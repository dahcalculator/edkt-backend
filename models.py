from sqlalchemy import Column, Integer, String, Float, ForeignKey
from database import Base





# This must be exactly 'User' with a capital U
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    matric_no = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="student")  # "student" or "admin"

# Keep your other models below it...
class Syllabus(Base):
    __tablename__ = "syllabus"
    id = Column(Integer, primary_key=True, index=True)
    topic_name = Column(String)

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("syllabus.id"))
    content = Column(String)
    option_a = Column(String)
    option_b = Column(String)
    option_c = Column(String)
    option_d = Column(String)
    correct_answer = Column(String)

class InteractionLog(Base):
    __tablename__ = "interaction_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    is_correct = Column(Integer)
    response_time = Column(Float)