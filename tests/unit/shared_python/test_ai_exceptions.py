from __future__ import annotations

from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    ScientificValidationError,
    ToolExecutionError,
    WorkflowError,
)


def test_aierror_init():
    err = AIError("test msg", details={"k": "v"})
    assert err.message == "test msg"
    assert err.details == {"k": "v"}


def test_aierror_str():
    err = AIError("test msg", details={"k": "v", "k2": "v2"})
    assert str(err) == "test msg (k=v, k2=v2)"

    err2 = AIError("only msg")
    assert str(err2) == "only msg"


def test_aiprovidererror():
    err = AIProviderError("prov msg", provider="test_prov", status_code=500, details={"ctx": 1})
    assert err.message == "prov msg"
    assert err.provider == "test_prov"
    assert err.status_code == 500
    assert err.details == {"ctx": 1}


def test_aiconnectionerror():
    err = AIConnectionError("conn msg", provider="p", status_code=503)
    assert isinstance(err, AIProviderError)
    assert err.message == "conn msg"
    assert err.status_code == 503


def test_airatelimiterror():
    err = AIRateLimitError("rate msg", provider="p", retry_after=1.5, details={"d": 2})
    assert err.message == "rate msg"
    assert err.status_code == 429
    assert err.retry_after == 1.5


def test_aitimeouterror():
    err = AITimeoutError("timeout", provider="p", timeout=10.0, details={"d": 3})
    assert err.message == "timeout"
    assert err.timeout == 10.0
    assert err.status_code is None


def test_scientificvalidationerror():
    err = ScientificValidationError(
        "sci_err", check_name="chk", value=1.0, threshold=0.5, details={"ctx": 4}
    )
    assert err.message == "sci_err"
    assert err.check_name == "chk"
    assert err.value == 1.0
    assert err.threshold == 0.5
    assert err.details == {"ctx": 4}


def test_workflowerror():
    err = WorkflowError("wf err", "wf123", step_id="step1", details={"c": 5})
    assert err.workflow_id == "wf123"
    assert err.step_id == "step1"
    assert err.message == "wf err"


def test_toolexecutionerror():
    err = ToolExecutionError(
        "tool fail", tool_name="my_tool", parameters={"p": 1}, details={"d": 1}
    )
    assert err.message == "tool fail"
    assert err.tool_name == "my_tool"
    assert err.parameters == {"p": 1}
    assert err.details == {"d": 1}

    err2 = ToolExecutionError("tool fail2", tool_name="tool2")
    assert err2.parameters == {}
