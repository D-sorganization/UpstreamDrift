"""Shared golf session service coordinating destination lifecycle and one-impact submission.

Follows TDD, DbC, Law of Demeter, and DRY.
All boundary invariants are verified regardless of python -O settings.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import TYPE_CHECKING, Any

from src.shared.python.golf_simulator.contracts import (
    CapabilityState,
    ConnectionStatus,
    ContactStatus,
    PreparedShot,
    SessionState,
    ShotEnvelope,
    SimulatorAdapter,
    SourceKind,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import DeliveryStatus

if TYPE_CHECKING:
    from src.shared.python.golf_simulator.journal import ShotJournal

logger = logging.getLogger(__name__)


class GolfSessionService:
    """Central session orchestrator for simulator destination selection and one-shot policy.

    Enforces:
    - Reconciled/idle state before destination switching (invalidates armed tokens).
    - Context revision matching between preparation and arming.
    - Single-use arm tokens consumed upon impact submission.
    - Multi-axis model qualification checks before arming.
    - Pre-send and post-send delivery journal auditing.
    - Explicit UNCERTAIN delivery handling and operator reconciliation.
    """

    def __init__(
        self,
        session_id: str,
        adapter: SimulatorAdapter,
        journal: ShotJournal | None = None,
    ) -> None:
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        self._session_id = str(session_id)
        self._adapter: SimulatorAdapter = adapter
        self._journal: ShotJournal | None = journal
        self._state: SessionState = SessionState.IDLE
        self._prepared_shot: PreparedShot | None = None
        self._receipts: dict[str, SubmissionReceipt] = {}
        self._destination_id: str = ""

    @property
    def current_state(self) -> SessionState:
        """Return the current lifecycle state of the session service."""
        return self._state

    @property
    def session_id(self) -> str:
        """Return the unique session identifier."""
        return self._session_id

    @property
    def current_destination_id(self) -> str:
        """Return the active destination identifier."""
        return self._destination_id

    async def select_destination(
        self,
        adapter: SimulatorAdapter,
        config: dict[str, Any] | None = None,
    ) -> ConnectionStatus:
        """Switch or connect the active simulator destination.

        Preconditions:
        - Must not be in ARMED or SUBMITTING state (no in-flight switching).
        - If PREPARED, switching destination invalidates any prepared shot and resets to IDLE.
        - Must not be in UNCERTAIN state (requires operator reconciliation first).
        """
        if self._state in (SessionState.ARMED, SessionState.SUBMITTING):
            raise RuntimeError(
                f"Cannot switch destination while in state '{self._state.value}'; "
                "must disarm or cancel first"
            )
        if self._state == SessionState.UNCERTAIN:
            raise RuntimeError(
                "Cannot switch destination while session is in UNCERTAIN state; "
                "resolve pending submission first"
            )

        # Invalidate any prepared shot upon destination change
        if self._state == SessionState.PREPARED:
            logger.info("Destination change invalidating prepared shot")
            self._prepared_shot = None
            self._state = SessionState.IDLE

        self._adapter = adapter
        status = await self._adapter.connect(config or {})
        self._destination_id = status.endpoint
        return status

    def prepare(
        self,
        shot: ShotEnvelope,
        context_revision: int = 1,
    ) -> PreparedShot:
        """Prepare a shot for arming, auditing capabilities and qualification.

        Preconditions:
        - Session must be in IDLE state.
        - Shot session_id must match active session.
        - Destination must declare SUPPORTED for shot_input.
        - MODEL_CONTACT shots must possess QUALIFIED contact qualification.
        """
        if self._state == SessionState.UNCERTAIN:
            raise RuntimeError(
                "Session is in UNCERTAIN state; resolve prior delivery before preparing shots"
            )
        if self._state != SessionState.IDLE:
            raise RuntimeError(
                f"Cannot prepare shot in state '{self._state.value}'; session must be IDLE"
            )
        if shot.session_id != self._session_id:
            raise ValueError(
                f"Shot session_id '{shot.session_id}' does not match service session_id '{self._session_id}'"
            )

        # Audit destination capabilities
        caps = self._adapter.capabilities()
        if caps.shot_input.state != CapabilityState.SUPPORTED:
            raise ValueError(
                f"Destination does not support shot input ({caps.shot_input.evidence})"
            )

        # Audit qualification requirements for model-driven shots
        if shot.source_kind == SourceKind.MODEL_CONTACT:
            qualification = shot.qualification
            contact_status = qualification.contact
            if contact_status != ContactStatus.QUALIFIED:
                status_name = contact_status.value
                raise ValueError(
                    f"Model shot requires qualified contact, got {status_name}"
                )

        prep_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prepared = PreparedShot(
            prepared_shot_id=prep_id,
            shot=shot,
            context_revision=context_revision,
            created_at_utc=now_utc,
            is_armed=False,
        )
        self._prepared_shot = prepared
        self._state = SessionState.PREPARED
        return prepared

    def arm(self, prepared_shot_id: str, context_revision: int) -> str:
        """Arm the prepared shot, verifying context revision and generating a single-use token.

        Preconditions:
        - Session must be in PREPARED state with matching prepared_shot_id.
        - context_revision must match the preparation revision.
        """
        if self._state != SessionState.PREPARED or self._prepared_shot is None:
            raise RuntimeError(
                f"No prepared shot to arm (current state: '{self._state.value}')"
            )
        if self._prepared_shot.prepared_shot_id != prepared_shot_id:
            raise ValueError("prepared_shot_id does not match active prepared shot")
        if self._prepared_shot.context_revision != context_revision:
            raise ValueError(
                f"Context revision mismatch: prepared for {self._prepared_shot.context_revision}, "
                f"got {context_revision}. Context was modified after preparation."
            )

        token = f"arm-{uuid.uuid4().hex}"
        self._prepared_shot = PreparedShot(
            prepared_shot_id=self._prepared_shot.prepared_shot_id,
            shot=self._prepared_shot.shot,
            context_revision=self._prepared_shot.context_revision,
            created_at_utc=self._prepared_shot.created_at_utc,
            arm_token=token,
            is_armed=True,
        )
        self._state = SessionState.ARMED
        return token

    def disarm(self, prepared_shot_id: str) -> None:
        """Disarm the current armed shot, discarding the arm token and returning to PREPARED."""
        if (
            self._prepared_shot is None
            or self._prepared_shot.prepared_shot_id != prepared_shot_id
        ):
            raise ValueError("prepared_shot_id does not match active prepared shot")

        if self._state == SessionState.ARMED:
            self._prepared_shot = PreparedShot(
                prepared_shot_id=self._prepared_shot.prepared_shot_id,
                shot=self._prepared_shot.shot,
                context_revision=self._prepared_shot.context_revision,
                created_at_utc=self._prepared_shot.created_at_utc,
                arm_token=None,
                is_armed=False,
            )
            self._state = SessionState.PREPARED

    def cancel(self, prepared_shot_id: str) -> None:
        """Cancel the prepared or armed shot, returning session to IDLE."""
        if (
            self._prepared_shot is None
            or self._prepared_shot.prepared_shot_id != prepared_shot_id
        ):
            raise ValueError("prepared_shot_id does not match active prepared shot")

        self._prepared_shot = None
        self._state = SessionState.IDLE

    async def submit_at_impact(
        self,
        prepared_shot_id: str,
        arm_token: str,
    ) -> SubmissionReceipt:
        """Execute one-shot submission to the simulator destination upon impact trigger.

        Preconditions:
        - Session must be ARMED.
        - arm_token and prepared_shot_id must match.
        """
        if self._state != SessionState.ARMED or self._prepared_shot is None:
            raise RuntimeError("Session is not armed; cannot submit at impact")
        if self._prepared_shot.prepared_shot_id != prepared_shot_id:
            raise ValueError("prepared_shot_id does not match active armed shot")
        if self._prepared_shot.arm_token != arm_token:
            raise ValueError("Invalid arm token")

        shot = self._prepared_shot.shot
        self._state = SessionState.SUBMITTING

        # 1. Pre-send intent logging
        if self._journal is not None:
            payload = f"{shot.shot_id}:{self._destination_id}".encode()
            self._journal.record_intent(
                shot_id=shot.shot_id,
                payload_bytes=payload,
            )

        # 2. Dispatch to destination adapter
        receipt = await self._adapter.submit(shot)

        # 3. Post-send audit logging
        if self._journal is not None:
            if receipt.state == SubmissionState.CONFIRMED_ACCEPTED:
                self._journal.record_acknowledgment(
                    shot_id=shot.shot_id,
                    code=200,
                    message=receipt.detail or "Delivered",
                )
            elif receipt.state == SubmissionState.UNKNOWN_AMBIGUOUS:
                self._journal.record_ambiguity(
                    shot_id=shot.shot_id,
                    reason=receipt.detail or "Ambiguous delivery",
                )
            else:
                self._journal.record_rejection(
                    shot_id=shot.shot_id,
                    code=400,
                    message=receipt.detail or "Rejected",
                )

        self._receipts[shot.shot_id] = receipt
        self._prepared_shot = None

        # 4. State transition
        if receipt.state == SubmissionState.UNKNOWN_AMBIGUOUS:
            self._state = SessionState.UNCERTAIN
        else:
            self._state = SessionState.IDLE

        return receipt

    def delivery_status(self, shot_id: str) -> SubmissionReceipt | None:
        """Query submission receipt for a delivered shot."""
        if shot_id in self._receipts:
            return self._receipts[shot_id]
        if self._journal is not None:
            entry = self._journal.get_entry(shot_id)
            if entry is not None:
                if entry.status == DeliveryStatus.ACKNOWLEDGED:
                    sub_state = SubmissionState.CONFIRMED_ACCEPTED
                elif entry.status == DeliveryStatus.AMBIGUOUS:
                    sub_state = SubmissionState.UNKNOWN_AMBIGUOUS
                elif entry.status == DeliveryStatus.REJECTED:
                    sub_state = SubmissionState.REJECTED
                else:
                    sub_state = SubmissionState.FAILED_BEFORE_SEND

                return SubmissionReceipt(
                    shot_id=entry.shot_id,
                    session_id=self._session_id,
                    state=sub_state,
                    destination_id=self._destination_id or "journal",
                    attempt_id=f"journal-{entry.shot_id}",
                    timestamp_utc=entry.updated_at_utc,
                    detail=entry.response_message or entry.error or "",
                )
        return None

    def resolve_uncertain(
        self,
        shot_id: str,
        operator_evidence: str,
        confirmed: bool,
    ) -> SubmissionReceipt:
        """Reconcile an uncertain submission via operator confirmation."""
        if self._state != SessionState.UNCERTAIN:
            raise RuntimeError(
                f"Session is in '{self._state.value}' state, not UNCERTAIN; cannot reconcile"
            )
        if not operator_evidence or not str(operator_evidence).strip():
            raise ValueError("operator_evidence must be a non-empty string")

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_state = (
            SubmissionState.CONFIRMED_ACCEPTED
            if confirmed
            else SubmissionState.REJECTED
        )
        reconciled = SubmissionReceipt(
            shot_id=shot_id,
            session_id=self._session_id,
            state=new_state,
            destination_id=self._destination_id or "reconciled",
            attempt_id=str(uuid.uuid4()),
            timestamp_utc=now_utc,
            detail=f"Operator reconciliation: {operator_evidence}",
        )
        self._receipts[shot_id] = reconciled

        if self._journal is not None:
            try:
                self._journal.get_entry(shot_id)
            except KeyError:
                self._journal.record_intent(
                    shot_id=shot_id,
                    payload_bytes=operator_evidence.encode("utf-8"),
                )
            if confirmed:
                self._journal.record_acknowledgment(
                    shot_id=shot_id,
                    code=200,
                    message=f"Operator reconciliation: {operator_evidence}",
                )
            else:
                self._journal.record_rejection(
                    shot_id=shot_id,
                    code=400,
                    message=f"Operator reconciliation: {operator_evidence}",
                )

        self._state = SessionState.IDLE
        return reconciled
