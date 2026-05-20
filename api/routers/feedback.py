"""
POST /v1/feedback — user correction endpoint.
No API key required — we want maximum feedback volume.
Logged to SQLite for future retraining.
"""

import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter
from api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/v1", tags=["Feedback"])
logger = logging.getLogger(__name__)

# Simple file-based log (upgrade to DB in production)
FEEDBACK_LOG = Path("data/feedback.jsonl")
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """
    Accept user correction for a prediction.
    Stored as JSONL for easy batch retraining ingestion.
    """
    import json
    entry = {
        "request_id":    body.request_id,
        "correct_label": body.correct_label,
        "comment":       body.comment,
        "submitted_at":  datetime.utcnow().isoformat(),
    }
    try:
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Feedback logged: {body.request_id} → {body.correct_label}")
        return FeedbackResponse(accepted=True,
                                message="Thank you. Feedback logged for retraining.")
    except Exception as e:
        logger.error(f"Feedback write failed: {e}")
        return FeedbackResponse(accepted=False, message="Failed to store feedback.")
