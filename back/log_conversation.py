
from sqlalchemy.orm import Session
from sqlalchemy import or_
from bdd.models import ConversationLog, Conversation

# def log_conversation(prompt, response):
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open("./log/conversation_logs.txt", "a") as log_file:
#         log_file.write(f"[{current_time}] Prompt: {prompt}\n")
#         log_file.write(f"[{current_time}] Response: {response}\n\n")


def log_conversation(db: Session, user_id: int, prompt: str, response: str):
    conversation_log = ConversationLog(user_id=user_id, prompt=prompt, response=response)
    db.add(conversation_log)
    db.commit()
    db.refresh(conversation_log)
    return conversation_log

def create_or_get_conversation(db: Session, user_id: int, category: str) -> Conversation:
    # Vérifier s'il existe déjà une conversation active dans la même catégorie
    existing_conversation = db.query(Conversation).filter(
        or_(
            (Conversation.user1_id == user_id) & (Conversation.user2_id == 1),
            (Conversation.user1_id == 1) & (Conversation.user2_id == user_id)
        ),
        Conversation.category == category
    ).first()

    if existing_conversation:
        # Vérifier si la conversation existante a un end_time défini
        if existing_conversation.end_time:
            # Si la conversation existante a un end_time défini, créer une nouvelle conversation
            new_conversation = Conversation(user1_id=user_id, user2_id=1, category=category)
            db.add(new_conversation)
            db.commit()
            db.refresh(new_conversation)
            return new_conversation
        else:
            # Si la conversation existante n'a pas d'end_time défini, retourner la conversation existante
            return existing_conversation
    else:
        # Créer une nouvelle conversation car aucune conversation active dans cette catégorie
        new_conversation = Conversation(user1_id=user_id, user2_id=1, category=category)
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        return new_conversation
    
