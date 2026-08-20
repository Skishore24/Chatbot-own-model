"""
backend/app/schemas/lead.py
----------------------------------------------------
Pydantic schemas for lead capture.
"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LeadCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Contact name")
    email: EmailStr = Field(..., description="Contact valid email address")
    phone: Optional[str] = Field(default=None, max_length=25, description="Contact phone number")
    message: Optional[str] = Field(default=None, max_length=2000, description="Project message or requirement")
    session_id: Optional[str] = Field(default=None, description="Associated chat session ID")


class LeadResponse(BaseModel):
    success: bool = True
    message: str = "Lead captured successfully"
    lead_id: Optional[int] = None
