import re
from src.guardrails import BoundedClinicalRetriever, GuardrailDecision
from src.rbac_scheduler import RBACAppointmentScheduler, SessionContext


class ClinicalFrontDeskPipeline:
    """Coordinates safety gates, bounded SOP retrieval, and RBAC scheduling."""

    def __init__(self, knowledge_base: list[dict]):
        self.retriever = BoundedClinicalRetriever(knowledge_base)
        self.scheduler = RBACAppointmentScheduler()

    def handle_request(self, query: str, session: SessionContext) -> GuardrailDecision:
        # 1. Always run adversarial & clinical safety gates first
        triggered, action, msg = self.retriever.check_deterministic_gates(query)
        if triggered:
            return GuardrailDecision(
                action=action, confidence=1.0, matched_doc_id=None, message=msg
            )

        # 2. Detect patient appointment lookup intent (e.g., PT-XXXX)
        patient_match = re.search(r"\b(PT-\d{4})\b", query.upper())
        if "appointment" in query.lower() or patient_match:
            target_id = (
                patient_match.group(1)
                if patient_match
                else session.authenticated_patient_id
            )
            status, record = self.scheduler.get_appointment(session, target_id)
            if status == "RBAC_DENIED":
                return GuardrailDecision(
                    action="RBAC_DENIED",
                    confidence=1.0,
                    matched_doc_id=None,
                    message=f"Access Denied: Session {session.authenticated_patient_id} is not authorized to view records for {target_id}.",
                )
            if status == "RBAC_SCHEDULER" and record:
                return GuardrailDecision(
                    action="RBAC_SCHEDULER",
                    confidence=1.0,
                    matched_doc_id=record["patient_id"],
                    message=f"Confirmed {record['clinic']} appointment on {record['date']} at {record['slot']}.",
                )

        # 3. Otherwise execute bounded SOP retrieval
        return self.retriever.retrieve(query)
