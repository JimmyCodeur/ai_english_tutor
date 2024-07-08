
from sqlalchemy.orm import Session
from sqlalchemy import or_
from bdd.models import ConversationLog, Conversation

def log_conversation(db: Session, user_id: int, prompt: str, response: str):
    conversation_log = ConversationLog(user_id=user_id, prompt=prompt, response=response)
    db.add(conversation_log)
    db.commit()
    db.refresh(conversation_log)
    return conversation_log

def create_or_get_conversation(db: Session, user_id: int, category: str) -> Conversation:
    existing_conversation = db.query(Conversation).filter(
        or_(
            (Conversation.user1_id == user_id) & (Conversation.user2_id == 1),
            (Conversation.user1_id == 1) & (Conversation.user2_id == user_id)
        ),
        Conversation.category == category
    ).first()

    if existing_conversation:
        if existing_conversation.end_time:
            new_conversation = Conversation(user1_id=user_id, user2_id=1, category=category)
            db.add(new_conversation)
            db.commit()
            db.refresh(new_conversation)
            return new_conversation
        else:
            return existing_conversation
    else:
        new_conversation = Conversation(user1_id=user_id, user2_id=1, category=category)
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        return new_conversation
    
