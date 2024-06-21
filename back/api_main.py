from fastapi import FastAPI, Depends, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from bdd.schema_pydantic import UserCreate, User, CreateAdminRequest as PydanticUser
from bdd.schema_pydantic import TokenInBody, UserLoginResponse
from pydantic import EmailStr, constr
from datetime import date
from bdd.database import get_db, engine
from bdd.crud import create_user, delete_user, update_user, get_user_by_id, update_user_role
from bdd.models import Base, User as DBUser, Role
from passlib.context import CryptContext
from token_security import get_current_user, authenticate_user, create_access_token, revoked_tokens

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="./front/static"), name="static")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get('/')
def index():
    return FileResponse('./front/form-login.html')

@app.get("/users-profile.html")
def get_users_profile():
    return FileResponse('./front/users-profile.html')

@app.get("/form-register.html")
def get_users_profile():
    return FileResponse('./front/form-register.html')

@app.get("/form-login.html")
def get_users_profile():
    return FileResponse('./front/form-login.html')

@app.get("/profile-setting.html")
def get_users_profile():
    return FileResponse('./front/profile-setting.html')

@app.get("/brain-info-course.html")
def get_users_profile():
    return FileResponse('./front/brain-info-course.html')

@app.get("/brain-course.html")
def get_users_profile():
    return FileResponse('./front/brain-course.html')

@app.get("/home.html")
def get_users_profile():
    return FileResponse('./front/home.html')

@app.get("/users/me", response_model=User)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users/{user_id}", response_model=PydanticUser)
def read_user_by_id(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/login/", response_model=UserLoginResponse)
def login_for_access_token(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    print(f"Generated Access Token: {access_token}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "nom": user.nom
    }


@app.post("/logout")
def logout(token_data: TokenInBody, db: Session = Depends(get_db)):
    token = token_data.token
    if token is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token missing")
    
    if token in revoked_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token already revoked")

    revoked_tokens.add(token)
    return {"message": "Successfully logged out"}


@app.post("/users/", response_model=User)
def create_user_endpoint(
    email: EmailStr = Form(...),
    nom: str = Form(...),
    date_naissance: date = Form(..., description="Format attendu : YYYY-MM-DD"),
    password: constr(min_length=6) = Form(..., description="Mot de passe de l'utilisateur (minimum 6 caractères)"),
    db: Session = Depends(get_db)
):
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance.isoformat(),
        password=password
    )
    try:
        db_user = create_user(db=db, user=user_data)
        return db_user
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

@app.post("/token", response_model=dict)
def login_for_access_token(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@app.put("/update-user-role/{user_id}")
def update_user_role_endpoint(
    user_id: int,
    new_role: str,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    if current_user.role != Role.ADMIN.name:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Insufficient permissions")
    
    updated = update_user_role(db=db, user_id=user_id, new_role=new_role)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur non trouvé")
    
    return {"message": f"Rôle de l'utilisateur {user_id} mis à jour avec succès"}

@app.put("/users/{user_id}")
def update_user_route(
    user_id: int,
    email: EmailStr = Form(None),
    nom: str = Form(None),
    date_naissance: date = Form(None),
    password: str = Form(None),
    db: Session = Depends(get_db)
):
    user_data = UserCreate(
        email=email,
        nom=nom,
        date_naissance=date_naissance,
        password=password
    )
    updated_user = update_user(db=db, user_id=user_id, user_data=user_data)
    return updated_user

@app.delete("/users/{user_id}")
def delete_user_route(user_id: int, db: Session = Depends(get_db)):
    deleted = delete_user(db=db, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return {"message": "Utilisateur supprimé avec succès"}

@app.post("/chat/")
async def chat_with_brain(audio_file: UploadFile = File(...)):
    # Save uploaded audio file
    audio_path = f"./uploads/{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Transcribe audio
    user_input = transcribe_audio(audio_path).strip()
    
    if not user_input:
        raise HTTPException(status_code=400, detail="Aucune entrée utilisateur détectée dans l'audio.")

    # Process user input and generate response
    prompt = (
        "As an English-speaking correspondent, your mission is to assist a French-speaking user in learning and enhancing their English skills through engaging and adaptive conversations.\n"
        # Your detailed prompt text here
        f"User: {user_input}\n"
    )
    
    generated_response = generate_ollama_response(model_name, prompt)
    
    if isinstance(generated_response, list):
        generated_response = ' '.join(generated_response)
    
    # Convert generated response to speech
    audio_file_path = text_to_speech_audio(generated_response)
    
    # Log conversation
    log_conversation(prompt, generated_response)
    
    # Play generated audio
    os.system(f"aplay {audio_file_path}")
    
    # Return response to frontend (optional)
    return {"user_input": user_input, "generated_response": generated_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
