"""
backend/app/api/leads.py
----------------------------------------------------
Lead capture endpoint for Genkit AI V6.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.lead import LeadCreateRequest, LeadResponse
from app.database.repository import LeadRepository
from app.core.logger import logger

router = APIRouter(tags=["Leads"])


@router.post("/leads", response_model=LeadResponse)
async def create_lead_endpoint(request: LeadCreateRequest):
    """Saves a user lead captured from the chat widget."""
    try:
        lead_id = LeadRepository.create_lead(
            name=request.name,
            email=str(request.email),
            phone=request.phone,
            message=request.message,
            session_id=request.session_id,
        )
        logger.info(f"Captured lead ID {lead_id} for email {request.email}")
        return LeadResponse(
            success=True,
            message="Thank you! Our Genkit team will reach out to you shortly.",
            lead_id=lead_id,
        )
    except Exception as e:
        logger.error(f"Failed to capture lead: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record lead. Please contact us directly at support@genkit.in",
        )


@router.get("/leads")
async def get_leads_endpoint(limit: int = 100):
    """Retrieves captured leads for administrative/dashboard viewing."""
    leads = LeadRepository.get_leads(limit=limit)
    return {"success": True, "count": len(leads), "leads": leads}


