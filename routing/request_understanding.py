import re
from base import Request, Context, Response, RequestProfile

# Domain keyword lists
FINANCE_KEYWORDS = [
    "finance", "financial", "revenue", "profit", "ebitda", "earnings",
    "stock", "stocks", "dividend", "dividends", "portfolio", "investment",
    "investments", "banking", "bank", "banks", "fiscal", "equity", "debt",
    "balance sheet", "quarterly", "10-k", "10-q", "sec filing", "sec filings", "hedge fund",
    "treasury", "valuation", "interest rate", "repo rate", "mortgage",
    "asset", "assets", "liability", "liabilities", "working capital",
    "venture capital", "capital gains", "crypto", "cryptocurrency",
    "nasdaq", "nyse", "bond", "bonds", "tax", "taxes",
    "expense ratio", "shareholder", "shareholders", "inflation", "market cap",
]

HEALTHCARE_KEYWORDS = [
    "healthcare", "health", "medical", "medicine", "medicines", "patient",
    "patients", "doctor", "physician", "hospital", "clinic", "clinical",
    "diagnosis", "symptom", "symptoms", "treatment", "prescription",
    "prescriptions", "prescribe", "drug", "drugs", "medication", "medications",
    "therapy", "disease", "illness", "surgery", "cardiology", "oncology",
    "nurse", "dosage", "pharmacy", "pathology", "vaccine", "vaccines",
    "vaccination", "paracetamol", "insulin", "diabetes", "pediatric",
    "hypertension", "infection", "therapeutic",
]

SUMMARIZATION_PATTERNS = [
    r"\b(summarize|summarise|summary|summaries|summarizing|summarising|summarization|summarisation|tl;?dr|synopsis|recap|overview|briefing|abstract)\b"
]

QA_PATTERNS = [
    r"\?",
    r"\b(what|who|where|when|why|how|which|whose|whom|explain|define|describe)\b",
    r"\b(is|are|can|could|does|do|did|should|would)\b",
]

CHAT_GREETINGS = [
    r"\b(hello|hi|hey|greetings|howdy|good morning|good afternoon|good evening)\b",
    r"\b(how are you|how's it going|hows it going|what's up|whats up|nice to meet you)\b",
    r"\b(bye|goodbye|see you|thanks|thank you|chat|chitchat)\b",
]


def determine_task_type(text: str | None) -> str | None:
    if not text or not text.strip():
        return None

    cleaned_text = text.strip()

    # 1. Summarization check
    for pattern in SUMMARIZATION_PATTERNS:
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            return "summarization"

    # 2. Check for chat greetings vs substantive QA question
    residual = cleaned_text
    for greeting_pat in CHAT_GREETINGS:
        residual = re.sub(greeting_pat, " ", residual, flags=re.IGNORECASE)
    residual_clean = re.sub(r"[?!.,;:\"']+", " ", residual).strip()

    residual_has_qa = any(re.search(pat, residual_clean, re.IGNORECASE) for pat in QA_PATTERNS)
    if residual_has_qa:
        return "qa"

    has_chat_indicator = any(re.search(pat, cleaned_text, re.IGNORECASE) for pat in CHAT_GREETINGS)
    if has_chat_indicator:
        return "chat"

    has_qa_indicator = any(re.search(pat, cleaned_text, re.IGNORECASE) for pat in QA_PATTERNS)
    if has_qa_indicator:
        return "qa"

    return None


def determine_domain(text: str | None) -> str:
    if not text or not text.strip():
        return "general"

    finance_count = sum(
        1 for kw in FINANCE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE)
    )
    healthcare_count = sum(
        1 for kw in HEALTHCARE_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE)
    )

    if finance_count == 0 and healthcare_count == 0:
        return "general"
    if finance_count >= healthcare_count:
        return "finance"
    return "healthcare"


def build_request_profile(request: Request, context: Context, response: Response) -> RequestProfile:
    task_type = determine_task_type(request.text if request else None)

    # Search for domain keywords across request, context, and response
    domain_corpus = " ".join(
        filter(
            None,
            ([request.text] if request and request.text else [])
            + [c.text for c in context if c and c.text]
            + ([response.text] if response and response.text else [])
        )
    )
    domain = determine_domain(domain_corpus)
    risk_level = "high" if domain in ("finance", "healthcare") else "low"
    context_available = len(context) > 0

    return RequestProfile(
        task_type=task_type,
        domain=domain,
        risk_level=risk_level,
        context_available=context_available,
    )

