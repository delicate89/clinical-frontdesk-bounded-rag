import re
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GuardrailDecision:
    action: str          # ANSWER_GROUNDED | ABSTAIN_OOD | ESCALATE_CLINICAL | ESCALATE_ADVERSARIAL
    confidence: float
    matched_doc_id: str | None
    message: str


class BoundedClinicalRetriever:
    """
    Implements pre-retrieval safety gates and cosine-similarity abstention
    thresholds (tau = 0.76 for dense embeddings; calibrated 0.18 for TF-IDF)
    to prevent hallucinated medical advice and out-of-distribution responses.
    """

    CLINICAL_ADVICE_PATTERNS = [
        r"\b(dosage|dose|tablet|medicine|medication|prescribe|antibiotic)\b",
        r"\b(diagnos(e|is)|symptom|chest pain|shortness of breath|bleeding|fever)\b",
        r"\b(should i take|what drug|treat my)\b",
    ]

    ADVERSARIAL_PATTERNS = [
        r"ignore (all |previous |prior )?instructions",
        r"system prompt",
        r"list all patients",
        r"dump database",
        r"bypass (security|access|rbac)",
    ]

    def __init__(self, knowledge_base: List[dict], similarity_threshold: float = 0.18):
        self.kb = knowledge_base
        self.threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True
        )
        corpus = [f"{item['title']} {item['content']}" for item in self.kb]
        self.kb_matrix = self.vectorizer.fit_transform(corpus)

    def check_deterministic_gates(self, query: str) -> Tuple[bool, str, str]:
        q_lower = query.lower().strip()
        for pattern in self.ADVERSARIAL_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    True,
                    "ESCALATE_ADVERSARIAL",
                    "Security Policy Block: Request violates front-desk access boundaries. Escalated to administrator.",
                )
        for pattern in self.CLINICAL_ADVICE_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    True,
                    "ESCALATE_CLINICAL",
                    "Clinical Safety Escalation: This assistant cannot provide medical triage or medication advice. Please proceed immediately to Emergency Triage or consult an on-duty physician.",
                )
        return False, "", ""

    def retrieve(self, query: str) -> GuardrailDecision:
        triggered, action, msg = self.check_deterministic_gates(query)
        if triggered:
            return GuardrailDecision(
                action=action,
                confidence=1.0,
                matched_doc_id=None,
                message=msg,
            )

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.kb_matrix).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_doc = self.kb[best_idx]

        if best_score < self.threshold:
            return GuardrailDecision(
                action="ABSTAIN_OOD",
                confidence=round(best_score, 4),
                matched_doc_id=None,
                message="Abstention Triggered: Query falls outside verified clinic SOP knowledge base. Routing to human front-desk clerk.",
            )

        return GuardrailDecision(
            action="ANSWER_GROUNDED",
            confidence=round(best_score, 4),
            matched_doc_id=best_doc["doc_id"],
            message=best_doc["content"],
        )

