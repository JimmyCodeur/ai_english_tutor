import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from schemas import UserCreate, User
from database import get_db, Base, engine
from crud import create_user

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.post("/users/", response_model=User)
def create_user_endpoint(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = create_user(db, user_data)
    return db_user

if __name__ == "__main__":
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
