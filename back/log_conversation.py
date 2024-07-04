
from sqlalchemy.orm import Session
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
    # Vérifier s'il existe déjà une conversation pour l'utilisateur courant et la catégorie donnée
    existing_conversation = db.query(Conversation).filter(
        (Conversation.user1_id == user_id) | (Conversation.user2_id == user_id),
        Conversation.category == category
    ).first()

    if existing_conversation:
        return existing_conversation
    else:
        # Créer une nouvelle conversation pour l'utilisateur courant et la catégorie donnée
        new_conversation = Conversation(user1_id=user_id, user2_id=1, category=category)  # Ici, 1 représente l'autre utilisateur
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        return new_conversation
