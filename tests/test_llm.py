import httpx
import pytest

from sqlagent.llm import LLM, LLMError


def make_llm(monkeypatch, handler, model="groq:some-model"):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = LLM(model)
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def ok(content="SELECT 1"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


def test_model_name_must_have_provider():
    with pytest.raises(ValueError):
        LLM("gpt-4o")


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLM("gemini:gemini-3.5-flash")


def test_counts_usage(monkeypatch):
    llm = make_llm(monkeypatch, lambda req: ok())
    assert llm.chat([{"role": "user", "content": "hi"}]) == "SELECT 1"
    assert llm.usage.calls == 1 and llm.usage.total_tokens == 12


def test_waits_on_429_then_succeeds(monkeypatch):
    replies = iter([httpx.Response(429, headers={"retry-after": "3"}), ok()])
    llm = make_llm(monkeypatch, lambda req: next(replies))
    assert llm.chat([]) == "SELECT 1"


def test_429_without_retry_after_backs_off_then_gives_up(monkeypatch):
    waits = []
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", waits.append)
    llm = LLM("groq:some-model")
    llm._http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(429)))
    with pytest.raises(LLMError) as e:
        llm.chat([])
    assert e.value.transient  # the benchmark skips the question instead of stopping
    assert waits[:5] == [10, 20, 40, 80, 120]


def test_connection_errors_are_retried(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("getaddrinfo failed")
        return ok()

    llm = make_llm(monkeypatch, handler)
    assert llm.chat([]) == "SELECT 1" and len(calls) == 3


def test_daily_limit_gives_up(monkeypatch):
    llm = make_llm(monkeypatch, lambda req: httpx.Response(429, headers={"retry-after": "7200"}))
    with pytest.raises(LLMError, match="daily limit"):
        llm.chat([])


def test_request_too_large_shrinks_reply_budget(monkeypatch):
    seen = []

    def handler(req):
        import json
        budget = json.loads(req.content)["max_tokens"]
        seen.append(budget)
        return ok() if budget <= 1024 else httpx.Response(413, json={"error": "Request too large"})

    llm = make_llm(monkeypatch, handler)
    assert llm.chat([]) == "SELECT 1"
    assert seen == [2048, 1024]


def test_retries_server_errors(monkeypatch):
    replies = iter([httpx.Response(503), httpx.Response(500), ok()])
    llm = make_llm(monkeypatch, lambda req: next(replies))
    assert llm.chat([]) == "SELECT 1"
