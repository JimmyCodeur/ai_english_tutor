from sqlalchemy.orm import Session
from sqlalchemy import or_
from back.bdd.models import ConversationLog, Message, Conversation

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
    
def log_conversation_to_db(db: Session, user_id: int, prompt: str, generated_response: str) -> None:
    log = ConversationLog(user_id=user_id, prompt=prompt, response=generated_response, user_audio_base64=None, user_input=None)
    db.add(log)
    db.commit()
    db.refresh(log)

def log_message_to_db(db: Session, user_id: int, conversation_id: int, prompt: str, generated_response: str, audio_base64: str, marker: str = None, ia_audio_duration: float = None) -> None:
    message = Message(user_id=user_id, conversation_id=conversation_id, content=prompt, user_input=None, user_audio_base64=None, ia_audio_base64=audio_base64, response=generated_response, marker=marker, ia_audio_duration=ia_audio_duration)
    db.add(message)
    db.commit()
    db.refresh(message)

def log_conversation_and_message(db: Session, user_id: int, category: str, current_prompt: str, user_input: str, generated_response: str, user_audio_base64: str, audio_base64: str, marker: str = None, suggestion: str = None, ia_audio_duration: float = None) -> None:
    conversation = db.query(Conversation).filter(
        (Conversation.user1_id == user_id),
        (Conversation.category == category),
        Conversation.active == True,
        Conversation.end_time == None
    ).first()

    if not conversation:
        conversation = create_or_get_conversation(db, user_id, category)
    
    log = ConversationLog(user_id=user_id, prompt=current_prompt, response=generated_response, user_audio_base64=audio_base64, user_input=user_input)
    db.add(log)
    db.commit()
    db.refresh(log)

    message = Message(user_id=user_id, conversation_id=conversation.id, content=current_prompt, ia_audio_base64=audio_base64, user_audio_base64=user_audio_base64, user_input=user_input, response=generated_response, marker=marker, suggestion=suggestion, ia_audio_duration=ia_audio_duration)
    db.add(message)
    db.commit()
    db.refresh(message)

    
