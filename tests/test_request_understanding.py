import pytest
from base import Request, ContextChunk, Response
from routing.request_understanding import build_request_profile


# ============================================================================
# 1. Tests for task_type: "qa" | "summarization" | "chat" | None
# (at least 3 examples per value)
# ============================================================================

@pytest.mark.parametrize("query", [
    "What was Apple's net profit in fiscal year 2024?",
    "Explain the physiological mechanism of insulin resistance.",
    "Where is the headquarters of the World Health Organization located?",
    "How does the Federal Reserve set the interest rate?",
])
def test_task_type_qa(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.task_type == "qa"


@pytest.mark.parametrize("query", [
    "Summarize the patient's discharge instructions and medication schedule.",
    "Please give me a concise summary of the Q3 10-Q filing.",
    "TLDR of the loan agreement terms and conditions.",
    "Can you provide a brief overview and recap of today's meeting?",
])
def test_task_type_summarization(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.task_type == "summarization"


@pytest.mark.parametrize("query", [
    "Hello there, how are you doing today?",
    "Good morning! Let's chat for a bit.",
    "Hi, thanks a lot for your help earlier!",
    "Hey! Nice to meet you.",
])
def test_task_type_chat(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.task_type == "chat"


@pytest.mark.parametrize("query", [
    "The quick brown fox jumps over the lazy dog.",
    "12345 67890 alpha beta gamma.",
    "Server reboot scheduled at 03:00 UTC.",
    "",
])
def test_task_type_none(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.task_type is None


# ============================================================================
# 2. Tests for domain: "finance" | "healthcare" | "general"
# (at least 3 examples per value)
# ============================================================================

@pytest.mark.parametrize("query", [
    "What is the company's annual revenue, EBITDA, and profit margin?",
    "Analyze the stock dividend payout and bond portfolio allocation.",
    "Review the quarterly balance sheet filed with the SEC in the 10-K.",
    "How does the mortgage interest rate impact treasury yields?",
])
def test_domain_finance(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.domain == "finance"
    assert profile.risk_level == "high"


@pytest.mark.parametrize("query", [
    "The patient was admitted to the hospital with acute pneumonia symptoms.",
    "What is the recommended dosage for paracetamol prescription in pediatrics?",
    "Clinical trials showed the vaccine reduced disease transmission by 90%.",
    "Doctor consultation notes indicate hypertension and type 2 diabetes treatment.",
])
def test_domain_healthcare(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.domain == "healthcare"
    assert profile.risk_level == "high"


@pytest.mark.parametrize("query", [
    "What is the capital city of Australia?",
    "Explain how gravity affects planetary orbits in astronomy.",
    "Write a Python function to sort a list using quicksort.",
    "Who was the architect of the Eiffel Tower?",
])
def test_domain_general(query):
    profile = build_request_profile(Request(text=query), [], Response(text=""))
    assert profile.domain == "general"
    assert profile.risk_level == "low"


# ============================================================================
# 3. Tests for risk_level: "high" | "low"
# (at least 3 examples per value)
# ============================================================================

@pytest.mark.parametrize("req_text, ctx_text", [
    ("Evaluate the corporate debt and equity ratio.", ""),
    ("Summarize clinical findings for oncology patients.", ""),
    ("Review the audit report.", "Audited balance sheet shows high liquidity and revenue growth."),
    ("Dosage guidelines for ibuprofen.", "Patient has severe headache in hospital clinic."),
])
def test_risk_level_high(req_text, ctx_text):
    ctx = [ContextChunk(text=ctx_text)] if ctx_text else []
    profile = build_request_profile(Request(text=req_text), ctx, Response(text=""))
    assert profile.risk_level == "high"
    assert profile.domain in ("finance", "healthcare")


@pytest.mark.parametrize("req_text", [
    "What is the recipe for chocolate chip cookies?",
    "Tell me the history of Mount Everest expeditions.",
    "How do airplanes stay in the air during flight?",
    "Good morning! Just dropping in to say hello.",
])
def test_risk_level_low(req_text):
    profile = build_request_profile(Request(text=req_text), [], Response(text=""))
    assert profile.risk_level == "low"
    assert profile.domain == "general"


# ============================================================================
# 4. Tests for context_available: True | False
# (at least 3 examples per value)
# ============================================================================

@pytest.mark.parametrize("ctx", [
    [ContextChunk(text="Chunk 1")],
    [ContextChunk(text="Doc A"), ContextChunk(text="Doc B")],
    [ContextChunk(text="Single factual statement.", source_id="src-1")],
])
def test_context_available_true(ctx):
    profile = build_request_profile(Request(text="test query"), ctx, Response(text=""))
    assert profile.context_available is True


@pytest.mark.parametrize("ctx", [
    [],
    [],
    [],
])
def test_context_available_false(ctx):
    profile = build_request_profile(Request(text="test query"), ctx, Response(text=""))
    assert profile.context_available is False
