"""Refuse to arm GitHub auto-merge on a pull request a reviewer has held back.

WHY THIS EXISTS
---------------
Fleet automation arms auto-merge using the repository owner's credentials, so
its arm is indistinguishable from a human's. That makes a reviewer's
``gh pr merge <n> --disable-auto`` unenforceable: the next automation pass just
re-arms it. Measured in ``Gasification_Model`` on 2026-08-14, every event
attributed to ``dieterolson`` (``type=User``, not a bot):

===== ================= ================= ======
PR    disarmed          re-armed          gap
===== ================= ================= ======
#4692 04:15:08Z         04:15:16Z         8s
#4710 04:15:39Z         04:15:45Z         6s
#4711 04:15:50Z         04:15:56Z         6s
#4709 03:38:35Z         04:02:09Z         24m
===== ================= ================= ======

PR #4709 is why this matters: it had auto-merge armed on a diff that DELETED 13
files present on ``main``. The reviewer's only remaining option was to close it.

Every fleet code path that arms auto-merge must go through :func:`arm_auto_merge`
rather than calling ``gh pr merge --auto`` directly. One seam, one policy.

This is the *arming* side of the fix. The enforcing side is the
``Merge-Hold-Guard.yml`` workflow deployed to each repo, which revokes arms this
module failed to prevent (for example, from an agent that shells out on its
own). Neither replaces the other: this one avoids the fight, the guard wins it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Labels that mean "a human said no". Case-insensitive.
#: ``do-not-automate`` is the pre-existing fleet-wide convention
#: (shared_scripts/agent_identity.DO_NOT_AUTOMATE_LABEL); honouring it here
#: keeps one vocabulary rather than a parallel one.
HOLD_LABELS = frozenset({"do-not-merge", "blocked", "do-not-automate"})

#: Label that acknowledges an intentional deletion of tracked files.
DELETIONS_ACK_LABEL = "deletions-acknowledged"

#: Body opt-out, for repos where the reviewer cannot apply labels.
DELETIONS_ACK_BODY = "deletions-acknowledged:"

#: Substring of the 403 body Claude Code cloud sessions get for any GraphQL call
#: (which ``gh pr merge --auto`` uses). Only this message selects the REST route.
GRAPHQL_BLOCKED_MARKER = "GitHub GraphQL is not available from Claude Code sessions"

#: Injected in tests so the whole module runs hermetically — no gh, no network.
CommandRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


#: GOV-1 (#1917): token prefixes. An App installation token (``ghs_``) acts as
#: the agent's bot identity; the others act as a human user, so agent actions
#: become indistinguishable from the owner's in audit logs and carry admin rights.
APP_TOKEN_PREFIXES = ("ghs_",)
USER_TOKEN_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_")

#: Environment variables gh consults, in gh's own precedence order.
TOKEN_ENV_VARS = ("GH_TOKEN", "GITHUB_TOKEN")

#: Exit code when ``--require-app-token`` refuses a non-App token.
EXIT_IDENTITY_REFUSED = 2

#: Exit code when an unverified REST arm could not be revoked (auto-merge may
#: still be ON). Distinct from 1 ("held / not armed") so automation can alert.
EXIT_ARM_MAY_BE_ON = 3


def _default_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and capture output. Resolved via PATH so Windows finds gh.exe."""
    exe = shutil.which(cmd[0]) or cmd[0]
    return subprocess.run(
        [exe, *cmd[1:]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def classify_token(token: str) -> str:
    """Classify a GitHub token by prefix without ever returning its value.

    Precondition: ``token`` is a ``str`` (empty means no token).
    Postcondition: returns one of ``"app"``, ``"user"``, ``"unknown"``, ``"none"``.
    """
    if not isinstance(token, str):
        raise TypeError("token must be a str")
    value = token.strip()
    if not value:
        return "none"
    if value.startswith(APP_TOKEN_PREFIXES):
        return "app"
    if value.startswith(USER_TOKEN_PREFIXES):
        return "user"
    return "unknown"


def active_token_kind(
    env: Mapping[str, str] | None = None, *, runner: CommandRunner = _default_runner
) -> str:
    """Return the kind of token gh will use: env vars first, then ``gh auth token``.

    The token value stays local to this function; only its kind is returned.
    """
    source = os.environ if env is None else env
    for var in TOKEN_ENV_VARS:
        if source.get(var, "").strip():
            return classify_token(source[var])
    try:
        proc = runner(["gh", "auth", "token"])
    except OSError:
        return "none"
    if proc.returncode != 0:
        return "none"
    return classify_token(proc.stdout or "")


_IDENTITY_WARNINGS = {
    "user": (
        "GOV-1: the active GitHub token is a personal/user token, not a GitHub "
        "App installation token (ghs_). Agent actions will be attributed to a "
        "human account. Run under the agent's App identity (#1917)."
    ),
    "unknown": (
        "GOV-1: the active GitHub token type is unrecognised; cannot confirm it "
        "is a GitHub App installation token (ghs_) (#1917)."
    ),
    "none": (
        "GOV-1: no GitHub token found (GH_TOKEN, GITHUB_TOKEN or gh auth); "
        "cannot confirm a GitHub App identity (#1917)."
    ),
}


def check_token_identity(
    *,
    require_app_token: bool,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner = _default_runner,
) -> bool:
    """Warn on stderr unless the active token is an App installation token.

    Returns ``False`` only when ``require_app_token`` is set and the token is not
    an App token; otherwise the warning is advisory and the caller proceeds.
    The token itself is never logged.
    """
    kind = active_token_kind(env, runner=runner)
    if kind == "app":
        return True
    logger.warning("WARNING: %s", _IDENTITY_WARNINGS[kind])
    return not require_app_token


@dataclass(frozen=True)
class HoldVerdict:
    """Why a pull request may not have auto-merge armed."""

    held: bool
    reasons: tuple[str, ...] = ()
    #: Populated on evaluation failure. A verdict that could not be computed is
    #: treated as HELD, so a broken token or a rate limit never opens the gate.
    error: str | None = None

    def describe(self) -> str:
        if not self.held:
            return "no hold in effect"
        return "; ".join(self.reasons) or "unknown hold"


@dataclass(frozen=True)
class ArmResult:
    """Outcome of an :func:`arm_auto_merge` call."""

    armed: bool
    verdict: HoldVerdict
    detail: str = ""
    #: True only when an unverified REST arm could not be revoked: GitHub may
    #: still auto-merge the PR. Callers must surface this loudly.
    auto_merge_may_be_on: bool = False

    def reason(self) -> str:
        """Why the PR is not armed: ``detail`` when set, else the hold verdict."""
        return self.detail or self.verdict.describe()


@dataclass
class _PullRequest:
    number: int
    draft: bool
    labels: list[str] = field(default_factory=list)
    body: str = ""
    head_sha: str = ""


def _gh_json(runner: CommandRunner, args: Sequence[str]) -> object:
    proc = runner(["gh", *args])
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    text = proc.stdout.strip()
    return json.loads(text) if text else None


def _gh_lines(runner: CommandRunner, args: Sequence[str]) -> list[str]:
    proc = runner(["gh", *args])
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _head_arrival(runner: CommandRunner, repo: str, pr: int, sha: str) -> str:
    """Return the GitHub-side time ``sha`` became the head of ``repo#pr``, or ``""``.

    Pre: ``sha`` is the head commit SHA of pull request ``pr`` in ``repo``.
    Post: an ISO-8601 UTC timestamp assigned by GitHub, or the empty string
    when none can be established.

    Why not the commit's ``committer.date``: the contributor writes it, so a
    future-dated commit would postdate any reviewer disarm and let automation
    re-arm at once. The earliest ``created_at`` among the check suites for the
    SHA is set by GitHub when the push lands and cannot be forged. (The PR
    timeline's ``committed`` events carry the same author-controlled date, and
    ``head_ref_force_pushed`` covers only force-pushes.) A later suite for the
    same SHA never moves the minimum.

    The suite time is when the SHA *first* got checks, which predates the PR if
    a branch is reset or force-pushed to a SHA seen elsewhere. So the result is
    ``max(earliest suite, latest head_ref_force_pushed event)``. If the timeline
    call fails the suite time alone is used: that can only over-hold, never
    re-arm early.
    """
    if not sha:
        return ""
    lines = _gh_lines(
        runner,
        [
            "api",
            f"repos/{repo}/commits/{sha}/check-suites?per_page=100",
            "--paginate",
            "--jq",
            ".check_suites[].created_at",
        ],
    )
    if not lines:
        return ""
    arrived = min(lines)
    try:
        pushes = _gh_lines(
            runner,
            [
                "api",
                f"repos/{repo}/issues/{pr}/timeline?per_page=100",
                "--paginate",
                "--jq",
                '.[] | select(.event == "head_ref_force_pushed") | .created_at',
            ],
        )
    except RuntimeError:
        return arrived
    return max([arrived, *pushes])


def evaluate_hold(
    repo: str, pr: int, *, runner: CommandRunner | None = None
) -> HoldVerdict:
    """Decide whether ``repo#pr`` is held back from auto-merge.

    ``repo`` is ``owner/name``. Signals mirror ``Merge-Hold-Guard.yml`` exactly,
    so the arming side and the enforcing side never disagree:

    1. a ``do-not-merge`` or ``blocked`` label
    2. draft state
    3. auto-merge disabled by a non-bot account more recently than the head
       commit — a reviewer said no and nobody has pushed since
    4. the diff deletes tracked files with no acknowledgement

    Fails closed: any error computing the verdict returns ``held=True``.
    """
    run = runner or _default_runner
    reasons: list[str] = []

    try:
        raw = _gh_json(
            run,
            ["api", f"repos/{repo}/pulls/{pr}"],
        )
        if not isinstance(raw, dict):
            return HoldVerdict(True, (), error=f"unreadable PR {repo}#{pr}")

        pull = _PullRequest(
            number=pr,
            draft=bool(raw.get("draft")),
            labels=[str(item.get("name", "")) for item in raw.get("labels") or []],
            body=str(raw.get("body") or ""),
            head_sha=str((raw.get("head") or {}).get("sha") or ""),
        )
        lowered = {label.lower() for label in pull.labels}

        # --- signal 1: explicit hold labels --------------------------------
        for label in sorted(lowered & HOLD_LABELS):
            reasons.append(f"`{label}` label")

        # --- signal 2: draft ------------------------------------------------
        if pull.draft:
            reasons.append("PR is a draft (mark ready first)")

        # --- signal 3: a reviewer disarmed it since the last push -----------
        # Bot actors are excluded so the Merge-Hold-Guard workflow's own
        # revocations never read as a human decision.
        disarms = _gh_lines(
            run,
            [
                "api",
                f"repos/{repo}/issues/{pr}/timeline?per_page=100",
                "--paginate",
                "--jq",
                '.[] | select(.event == "auto_merge_disabled") '
                '| select((.actor.type // "User") != "Bot") | .created_at',
            ],
        )
        arrived = _head_arrival(run, repo, pr, pull.head_sha) if disarms else ""
        if disarms:
            last_disarm = max(disarms)
            if not arrived:
                reasons.append(
                    f"a reviewer disabled auto-merge at {last_disarm} and the "
                    "server-side arrival time of the head commit is unknown"
                )
            elif last_disarm > arrived:
                reasons.append(
                    f"a reviewer disabled auto-merge at {last_disarm}, after the "
                    f"head commit arrived ({arrived}) — no push has superseded it"
                )

        # --- signal 4: unacknowledged deletion of tracked files -------------
        removed = _gh_lines(
            run,
            [
                "api",
                f"repos/{repo}/pulls/{pr}/files?per_page=100",
                "--paginate",
                "--jq",
                '.[] | select(.status == "removed") | .filename',
            ],
        )
        if removed and not _deletions_acknowledged(lowered, pull.body):
            sample = ", ".join(removed[:5])
            more = f" (+{len(removed) - 5} more)" if len(removed) > 5 else ""
            reasons.append(
                f"diff deletes {len(removed)} tracked file(s) with no "
                f"acknowledgement: {sample}{more}"
            )
    except (RuntimeError, json.JSONDecodeError, OSError) as exc:
        # Fail closed. An unknown state is not a licence to arm.
        return HoldVerdict(True, (), error=str(exc))

    return HoldVerdict(bool(reasons), tuple(reasons))


def _deletions_acknowledged(lowered_labels: set[str], body: str) -> bool:
    if DELETIONS_ACK_LABEL in lowered_labels:
        return True
    for line in body.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(DELETIONS_ACK_BODY):
            value = stripped[len(DELETIONS_ACK_BODY) :].strip()
            if value in {"yes", "true"}:
                return True
    return False


def _stored_merge_method(repo: str, pr: int, run: CommandRunner) -> str:
    """Return the auto-merge method GitHub actually stored on ``repo#pr``.

    Used after the REST fallback, whose route may not honour the requested
    method (Codex review on #1937). Returns "" when it cannot be read.
    """
    proc = run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}", "--jq", ".auto_merge.merge_method"]
    )
    return proc.stdout.strip().lower() if proc.returncode == 0 else ""


def _revoke_rest_arm(repo: str, pr: int, run: CommandRunner) -> tuple[str, bool]:
    """Undo a REST arm whose stored method could not be confirmed.

    The PUT already succeeded, so GitHub may still auto-merge the PR with the
    wrong strategy. Returns a message for ``ArmResult.detail``; a failed DELETE
    is reported loudly because auto-merge may then still be on. The second item
    is True in exactly that case.
    """
    proc = run(["gh", "api", "-X", "DELETE", f"repos/{repo}/pulls/{pr}/ccr/auto_merge"])
    if proc.returncode == 0:
        logger.warning("Revoked unverified REST auto-merge arm on %s#%s.", repo, pr)
        return "auto-merge revoked", False
    err = proc.stderr.strip() or proc.stdout.strip()
    logger.error("Could not revoke REST auto-merge arm on %s#%s: %s", repo, pr, err)
    return (
        f"revoke FAILED ({err}); auto-merge may still be ON and needs manual disarm",
        True,
    )


def arm_auto_merge(
    repo: str,
    pr: int,
    *,
    strategy: str = "squash",
    delete_branch: bool = False,
    runner: CommandRunner | None = None,
) -> ArmResult:
    """Arm auto-merge on ``repo#pr`` unless a reviewer has held it back.

    This is the ONLY sanctioned way for fleet automation to arm auto-merge.
    Calling ``gh pr merge --auto`` directly bypasses the reviewer's decision and
    is what this module exists to stop.

    Preconditions: ``repo`` is ``owner/name``; ``strategy`` is squash, merge or
    rebase. Postcondition: no arm call of any kind (GraphQL or REST) is made
    unless :func:`evaluate_hold` returned not-held. If the GraphQL arm fails
    with the Claude Code cloud "GraphQL is not available" 403 (and only that),
    exactly one fallback ``PUT repos/{repo}/pulls/{pr}/ccr/auto_merge`` is made,
    and it is reported as armed only if the merge method GitHub stored equals
    ``strategy``; otherwise the arm is revoked with a ``DELETE`` on the same
    route, and a failed revoke is reported in the detail.
    """
    run = runner or _default_runner
    verdict = evaluate_hold(repo, pr, runner=run)
    if verdict.held:
        logger.warning(
            "Refusing to arm auto-merge on %s#%s: %s", repo, pr, verdict.describe()
        )
        return ArmResult(False, verdict, "held")

    cmd = ["gh", "pr", "merge", str(pr), "--repo", repo, f"--{strategy}", "--auto"]
    if delete_branch:
        cmd.append("--delete-branch")
    proc = run(cmd)
    if proc.returncode != 0 and GRAPHQL_BLOCKED_MARKER in (proc.stderr + proc.stdout):
        logger.info("GraphQL blocked; arming %s#%s via the REST route.", repo, pr)
        proc = run(
            [
                "gh",
                "api",
                "-X",
                "PUT",
                f"repos/{repo}/pulls/{pr}/ccr/auto_merge",
                "-f",
                f"merge_method={strategy}",
            ]
        )
        if proc.returncode == 0:
            stored = _stored_merge_method(repo, pr, run)
            if stored != strategy:
                detail = (
                    f"REST route stored merge method {stored or 'none'!r}, "
                    f"not the requested {strategy!r}"
                )
                logger.warning("Not armed as requested on %s#%s: %s", repo, pr, detail)
                revoked, may_be_on = _revoke_rest_arm(repo, pr, run)
                full = f"{detail}; {revoked}"
                if may_be_on:
                    logger.error("DANGER %s#%s: %s", repo, pr, full)
                return ArmResult(False, verdict, full, auto_merge_may_be_on=may_be_on)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        logger.warning("Could not arm auto-merge on %s#%s: %s", repo, pr, detail)
        return ArmResult(False, verdict, detail)

    logger.info("Auto-merge armed on %s#%s (%s).", repo, pr, strategy)
    return ArmResult(True, verdict, "armed")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``automerge_guard.py <owner/repo> <pr> [--arm] [--strategy squash]``.

    Without ``--arm`` this only reports. Exit code 0 means "safe to arm", 1
    means held (or, with ``--arm``, that arming did not happen), 2 means
    ``--require-app-token`` refused a non-App token (GOV-1, #1917), 3 means
    an unverified REST arm could not be revoked (auto-merge may still be ON).
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="owner/name")
    parser.add_argument("pr", type=int)
    parser.add_argument("--arm", action="store_true", help="arm auto-merge if not held")
    parser.add_argument(
        "--strategy", default="squash", choices=["squash", "merge", "rebase"]
    )
    parser.add_argument("--delete-branch", action="store_true")
    parser.add_argument(
        "--require-app-token",
        action="store_true",
        help="exit 2 unless the active token is a GitHub App installation token",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not check_token_identity(require_app_token=args.require_app_token):
        return EXIT_IDENTITY_REFUSED

    if args.arm:
        result = arm_auto_merge(
            args.repo,
            args.pr,
            strategy=args.strategy,
            delete_branch=args.delete_branch,
        )
        if result.armed:
            print(f"armed {args.repo}#{args.pr}")
            return 0
        if result.auto_merge_may_be_on:
            print(f"DANGER NOT armed {args.repo}#{args.pr}: {result.reason()}")
            return EXIT_ARM_MAY_BE_ON
        print(f"NOT armed {args.repo}#{args.pr}: {result.reason()}")
        return 1

    verdict = evaluate_hold(args.repo, args.pr)
    print(f"{args.repo}#{args.pr}: {verdict.describe()}")
    return 1 if verdict.held else 0


if __name__ == "__main__":
    raise SystemExit(main())
