from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None

class ConversationCreate(BaseModel):
    title: str = "Nouvelle discussion"

class MessageCreate(BaseModel):
    role: str
    content: str
