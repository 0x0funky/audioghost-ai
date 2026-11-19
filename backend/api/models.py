"""
The SAM-Audio models: Meta's SAM License, which comes with the weights.

AudioGhost downloads its own public copies of the weights (no Hugging Face
account needed), and asks for the license once before the first download.
"""
from fastapi import APIRouter
from pydantic import BaseModel

import model_license

router = APIRouter()


class LicenseInfo(BaseModel):
    version: str
    url: str
    accepted: bool
    text: str


@router.get("/license", response_model=LicenseInfo)
def get_license():
    """The license text and whether this installation accepted it."""
    return LicenseInfo(version=model_license.VERSION, url=model_license.URL,
                       accepted=model_license.accepted(), text=model_license.text())


@router.post("/license/accept")
def accept_license():
    model_license.accept()
    return {"accepted": True, "version": model_license.VERSION}
