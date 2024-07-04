
from sqlalchemy.orm import Session
from bdd.models import ConversationLog

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