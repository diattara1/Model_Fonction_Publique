# rag_system/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import SessionLocal, User, Conversation, Message

router = APIRouter()

# === DB Dependency ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === Users ===
@router.post("/users/", response_model=dict)
def create_user(payload: dict, db: Session = Depends(get_db)):
    username = payload.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Nom d’utilisateur requis.")
    
    user = User(username=username)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except:
        db.rollback()
        raise HTTPException(status_code=400, detail="Nom d’utilisateur déjà pris.")
    return {"id": user.id, "username": user.username}

@router.get("/users/{user_id}", response_model=dict)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return {"id": user.id, "username": user.username}

@router.get("/users/byname/{username}", response_model=dict)
def get_user_byname(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return {"id": user.id, "username": user.username}

# === Conversations ===
@router.post("/users/{user_id}/conversations/", response_model=dict)
def create_conversation(user_id: int, title: str = "Nouvelle discussion", db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    conv = Conversation(user_id=user_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": conv.id, "title": conv.title, "created_at": conv.created_at}

@router.get("/users/{user_id}/conversations/", response_model=list)
def list_conversations(user_id: int, db: Session = Depends(get_db)):
    convs = db.query(Conversation).filter(Conversation.user_id == user_id).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in convs]

@router.get("/conversations/{conv_id}/messages", response_model=list)
def get_messages(conv_id: int, db: Session = Depends(get_db)):
    msgs = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.created_at).all()
    return [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at} for m in msgs]

# === Messages ===
@router.post("/conversations/{conv_id}/messages", response_model=dict)
def add_message(conv_id: int, role: str, content: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    msg = Message(conversation_id=conv_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "role": msg.role, "content": msg.content, "created_at": msg.created_at}
