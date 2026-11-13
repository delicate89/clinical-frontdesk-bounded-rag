from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SessionContext:
    session_id: str
    authenticated_patient_id: str
    role: str = "PATIENT"  # PATIENT | DESK_CLERK


class RBACAppointmentScheduler:
    """
    Enforces session-scoped Role-Based Access Control (RBAC) for appointment
    lookups so generative workflows cannot leak cross-patient schedules.
    """

    def __init__(self):
        self._appointments: Dict[str, dict] = {
            "PT-1042": {
                "patient_id": "PT-1042",
                "clinic": "Internal Medicine OPD",
                "date": "2026-08-18",
                "slot": "09:30 AM",
                "status": "CONFIRMED",
            },
            "PT-9999": {
                "patient_id": "PT-9999",
                "clinic": "Cardiology OPD",
                "date": "2026-08-19",
                "slot": "11:00 AM",
                "status": "CONFIRMED",
            },
        }

    def get_appointment(
        self, session: SessionContext, requested_patient_id: str
    ) -> tuple[str, Optional[dict]]:
        if (
            session.role == "PATIENT"
            and session.authenticated_patient_id != requested_patient_id
        ):
            return (
                "RBAC_DENIED",
                None,
            )

        record = self._appointments.get(requested_patient_id)
        if not record:
            return ("NOT_FOUND", None)

        return ("RBAC_SCHEDULER", record)
