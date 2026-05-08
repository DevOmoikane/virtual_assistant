from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from virtual_assistant_be.services.telegram_service import TelegramService

router = APIRouter(prefix="/api/telegram", tags=["telegram"])

telegram = TelegramService()


class AddContactRequest(BaseModel):
    name: str
    username: str = ""
    chat_id: int = 0


class SendMessageRequest(BaseModel):
    contact: str
    message: str


@router.get("/contacts")
async def list_contacts():
    return {"contacts": telegram.list_contacts()}


@router.post("/contacts")
async def add_contact(req: AddContactRequest):
    if not req.name.strip():
        raise HTTPException(400, "name is required")
    contact = telegram.add_contact(req.name.strip(), req.username.strip(), req.chat_id)
    return {"contact": contact}


@router.delete("/contacts/{name}")
async def delete_contact(name: str):
    if not telegram.remove_contact(name):
        raise HTTPException(404, f"Contact '{name}' not found")
    return {"deleted": name}


@router.post("/send")
async def send_message(req: SendMessageRequest):
    if not req.contact.strip() or not req.message.strip():
        raise HTTPException(400, "contact and message are required")
    result = telegram.send_message_to_contact(req.contact.strip(), req.message.strip())
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "send failed"))
    return result
