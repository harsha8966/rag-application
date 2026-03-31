"""API route modules."""

from app.api.routes.upload import router as upload_router
from app.api.routes.ask import router as ask_router
from app.api.routes.feedback import router as feedback_router

__all__ = ["upload_router", "ask_router", "feedback_router"]
