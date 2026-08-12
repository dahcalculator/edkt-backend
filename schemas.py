from pydantic import BaseModel

class UserCreate(BaseModel):
    fullName: str
    matric: str
    password: str

class UserLogin(BaseModel):
    matric: str
    password: str

class InteractionCreate(BaseModel):
    skill_id: int
    is_correct: int
    response_time: float

class InteractionSubmit(BaseModel):
    question_id: int
    is_correct: bool
    response_time: float # Captured in seconds

class InteractionSubmit(BaseModel):
    question_id: int
    is_correct: bool
    response_time: float # Captured in seconds

class InteractionSubmit(BaseModel):
    matric: str
    question_id: int
    is_correct: bool
    response_time: float # Time in seconds (e.g., 14.52)