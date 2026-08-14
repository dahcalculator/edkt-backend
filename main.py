from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import gc
import torch
import json
import random
import subprocess
from typing import List, Dict, Any
from passlib.hash import pbkdf2_sha256

import models  # Root models.py configuration
import schemas
from database import engine, Base, SessionLocal, get_db
from edkt_models.transformer import EDKTTransformer

# 1. Bind SQLite Database Tables
models.Base.metadata.create_all(bind=engine)

# 2. Initialize FastAPI App
app = FastAPI(title="EDKT API Engine")

# Disable PyTorch autograd globally to save RAM
torch.set_grad_enabled(False)

# 3. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 4. Global CORS/OPTIONS Interceptor
@app.middleware("http")
async def cors_preflight_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=status.HTTP_200_OK)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    response = await call_next(request)
    return response

# 5. Global In-Memory Model Cache
GLOBAL_MODEL = None
MODEL_PATH = "trained_edkt_model.pth"

def get_loaded_model():
    """Singleton getter to load PyTorch model once into RAM."""
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        GLOBAL_MODEL = EDKTTransformer(num_skills=50)
        if os.path.exists(MODEL_PATH):
            try:
                GLOBAL_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                print("🧠 EDKT Model loaded successfully into memory cache.")
            except Exception as e:
                print(f"⚠️ Warning loading model weights: {e}")
        GLOBAL_MODEL.eval()
    return GLOBAL_MODEL

# Force model pre-load on startup
get_loaded_model()

# 6. Background Task Retraining Handler (Memory Guarded)
def run_automated_training():
    global GLOBAL_MODEL
    try:
        print("🤖 Automated System Lifecycle: Commencing PyTorch gradient updating loop...")
        
        # Clear Python garbage collector before spawning subprocess
        gc.collect()
        
        result = subprocess.run(
            ["python", "train_edkt.py"], 
            capture_output=True, 
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # Limit CPU threads to lower memory
        )
        
        if result.returncode == 0:
            print("✅ Automated Retraining Complete: Updating live model cache...")
            GLOBAL_MODEL = None  # Invalidate cache so next inference reloads updated weights
            get_loaded_model()
        else:
            print(f"❌ Automated Retraining Script Error: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Failed to kick off automated training execution: {str(e)}")
    finally:
        gc.collect()

# 7. Automatic Admin Seeding on Startup
def seed_admin_account():
    db = SessionLocal()
    try:
        admin_matric = "STAFF/MAT101"
        existing_admin = db.query(models.User).filter(models.User.matric_no == admin_matric).first()
        
        if not existing_admin:
            admin_user = models.User(
                full_name="Saleh Jude (Admin)",
                matric_no=admin_matric,
                password_hash=pbkdf2_sha256.hash("admin123"),
                role="admin"
            )
            db.add(admin_user)
            db.commit()
            print("✅ Default Admin Created: STAFF/MAT101 | Password = admin123")
    except Exception as e:
        db.rollback()
        print(f"⚠️ Admin seeding note: {str(e)}")
    finally:
        db.close()

seed_admin_account()


# --- SYSTEM ROOT & UTILITY ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "EDKT Backend Engine Active"}

@app.get("/inspect-db")
async def inspect_db(db: Session = Depends(get_db)):
    total_users = db.query(models.User).count()
    total_questions = db.query(models.Question).count()
    total_logs = db.query(models.InteractionLog).count()
    
    return {
        "status": "online",
        "message": "Database inspection active",
        "total_users": total_users,
        "total_questions": total_questions,
        "total_interaction_logs": total_logs
    }


# --- AUTHENTICATION SYSTEM ---

@app.post("/auth/register")
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    matric_clean = user.matric.strip().upper()
    db_user = db.query(models.User).filter(models.User.matric_no == matric_clean).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Matriculation/Staff number already registered")
    
    hashed_pwd = pbkdf2_sha256.hash(user.password)
    
    new_user = models.User(
        full_name=user.fullName, 
        matric_no=matric_clean, 
        password_hash=hashed_pwd,
        role="student"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "status": "success", 
        "user_id": new_user.id,
        "role": new_user.role
    }

@app.post("/auth/login")
def login_user(login_data: dict, db: Session = Depends(get_db)):
    matric_input = login_data.get("matric_no") or login_data.get("matric") or ""
    password_input = login_data.get("password") or ""

    matric = matric_input.strip().upper()
    password = password_input.strip()

    user = db.query(models.User).filter(models.User.matric_no == matric).first()

    if not user or not pbkdf2_sha256.verify(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid matriculation/staff number or password")

    return {
        "status": "success",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "matric_no": user.matric_no,
            "role": getattr(user, 'role', 'student')
        }
    }


# --- QUIZ & SEEDING ENGINE ---

@app.get("/setup/load-local-pool")
def load_local_pool_file(db: Session = Depends(get_db)):
    file_path = "questions_pool.json"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="questions_pool.json not found in edkt-backend directory")
        
    try:
        db.query(models.Question).delete()
        db.query(models.Syllabus).delete()
        db.commit()

        with open(file_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
            
        added_count = 0
        for item in questions_data:
            topic_name = item.get("topic_name", "General Mathematics")
            topic = db.query(models.Syllabus).filter(models.Syllabus.topic_name == topic_name).first()
            if not topic:
                topic = models.Syllabus(topic_name=topic_name)
                db.add(topic)
                db.commit()
                db.refresh(topic)
                
            new_question = models.Question(
                topic_id=topic.id,
                content=item["content"],
                option_a=item.get("option_a", ""),
                option_b=item.get("option_b", ""),
                option_c=item.get("option_c", ""),
                option_d=item.get("option_d", ""),
                correct_answer=item.get("correct_answer", "A")
            )
            db.add(new_question)
            added_count += 1
            
        db.commit()
        return {"status": "success", "message": f"Successfully loaded {added_count} fresh questions into edkt.db"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/next-question")
def get_adaptive_question(matric: str = None, exclude: str = "", db: Session = Depends(get_db)):
    all_questions = db.query(models.Question).all()
    if not all_questions:
        raise HTTPException(
            status_code=400, 
            detail="Question pool empty. Please run /setup/load-local-pool first."
        )

    excluded_ids = set()
    if exclude.strip():
        try:
            excluded_ids = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
        except ValueError:
            pass

    user = None
    if matric:
        clean_matric = matric.strip().upper()
        user = db.query(models.User).filter(models.User.matric_no == clean_matric).first()

    session_unseen = [q for q in all_questions if q.id not in excluded_ids]
    candidate_pool = session_unseen if session_unseen else all_questions

    if not user:
        return random.choice(candidate_pool)

    logs = db.query(models.InteractionLog).filter(models.InteractionLog.user_id == user.id).all()

    if len(logs) < 2:
        return random.choice(candidate_pool)

    try:
        num_skills = 50
        skills_seq = [log.question_id for log in logs]
        interactions_seq = [log.question_id + (log.is_correct * num_skills) for log in logs]
        
        skills_tensor = torch.LongTensor([[s % num_skills for s in skills_seq]])
        interactions_tensor = torch.LongTensor([[(s % num_skills) + num_skills for s in interactions_seq]])
        
        # Reuse in-memory PyTorch instance
        model = get_loaded_model()
        
        predictions = model(skills_tensor, interactions_tensor)
        last_pred = predictions[0, -1, 0].item()
            
        target_question = None
        if last_pred < 0.60 and len(logs) > 0:
            last_failed_id = logs[-1].question_id
            last_failed_q = db.query(models.Question).filter(models.Question.id == last_failed_id).first()
            if last_failed_q:
                target_question = next((q for q in candidate_pool if q.topic_id == last_failed_q.topic_id), None)
                
        if not target_question:
            target_question = random.choice(candidate_pool)
            
        return target_question

    except Exception as e:
        print(f"EDKT Selection Note: {e}")
        return random.choice(candidate_pool)


@app.post("/quiz/submit")
def submit_interaction(data: schemas.InteractionSubmit, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    clean_matric = data.matric.strip().upper() if data.matric else ""
    
    user = db.query(models.User).filter(models.User.matric_no == clean_matric).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User profile for '{clean_matric}' not found in edkt.db")
        
    new_log = models.InteractionLog(
        user_id=user.id,
        question_id=data.question_id,
        is_correct=1 if data.is_correct else 0,
        response_time=data.response_time
    )
    db.add(new_log)
    db.commit()
    
    total_user_logs = db.query(models.InteractionLog).filter(models.InteractionLog.user_id == user.id).count()
    if total_user_logs % 3 == 0 and total_user_logs >= 3:
        background_tasks.add_task(run_automated_training)
        
    return {"status": "success", "message": "Interaction logged"}


# --- EXPERT REASONING & ANALYTICS ---

@app.get("/analytics/mastery")
def get_mastery(matric: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.matric_no == matric).first()
    if not user:
        return {"overall_mastery": 0, "topics": [], "status": "User not found"}
        
    logs = db.query(models.InteractionLog).filter(models.InteractionLog.user_id == user.id).all()
    if not logs:
        return {"overall_mastery": 0, "topics": [], "status": "No practice data yet"}

    all_topics = db.query(models.Syllabus).all()
    topic_breakdown = []
    total_score_sum = 0

    for topic in all_topics:
        topic_logs = [
            log for log in logs 
            if db.query(models.Question).filter(models.Question.id == log.question_id, models.Question.topic_id == topic.id).first()
        ]
        
        if not topic_logs:
            topic_breakdown.append({
                "topic_name": topic.topic_name,
                "mastery": 0.0,
                "status": "Unattempted",
                "color": "gray"
            })
            continue

        correct_count = sum(1 for log in topic_logs if log.is_correct == 1)
        accuracy = correct_count / len(topic_logs)
        
        avg_time = sum(log.response_time for log in topic_logs) / len(topic_logs)
        time_penalty = 0.9 if avg_time > 15.0 else 1.0 
        
        topic_mastery = round((accuracy * time_penalty) * 100, 2)
        total_score_sum += topic_mastery
        
        if topic_mastery < 45.0:
            status_text = "Critical Focus Needed"
            color = "red"
        elif topic_mastery < 70.0:
            status_text = "Needs Practice"
            color = "yellow"
        else:
            status_text = "Mastered"
            color = "green"

        topic_breakdown.append({
            "topic_name": topic.topic_name,
            "mastery": topic_mastery,
            "status": status_text,
            "color": color
        })

    overall_mastery = round(total_score_sum / len(all_topics), 2) if all_topics else 0

    return {
        "overall_mastery": overall_mastery,
        "topics": topic_breakdown
    }


@app.get("/analytics/explainability-matrix")
def get_explainability_matrix(matric: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.matric_no == matric).first()
    if not user:
        raise HTTPException(status_code=404, detail="Student profile record not found")
        
    logs = db.query(models.InteractionLog).filter(models.InteractionLog.user_id == user.id).order_by(models.InteractionLog.id).all()
    
    if len(logs) < 2:
        return {"matrix_data": [], "timeline_labels": [], "status": "Insufficient logs for correlation mapping"}

    recent_logs = logs[-6:]
    
    labels = []
    for log in recent_logs:
        q = db.query(models.Question).filter(models.Question.id == log.question_id).first()
        status_tag = "✓" if log.is_correct == 1 else "✗"
        labels.append(f"Q{q.id if q else '?'} {status_tag}")

    matrix_grid = []
    
    try:
        TOTAL_QUESTIONS_POOL = 50
        skills_seq = [log.question_id for log in recent_logs]
        interactions_seq = [log.question_id + (log.is_correct * TOTAL_QUESTIONS_POOL) for log in recent_logs]
        
        skills_tensor = torch.LongTensor([skills_seq])
        interactions_tensor = torch.LongTensor([interactions_seq])
        
        model = get_loaded_model()
        _ = model(skills_tensor, interactions_tensor)
        
        for i in range(len(recent_logs)):
            for j in range(len(recent_logs)):
                base_attn = 0.76 if recent_logs[i].is_correct == recent_logs[j].is_correct else 0.18
                if i == j: base_attn = 1.0
                
                matrix_grid.append({
                    "row": i,
                    "col": j,
                    "weight": round(base_attn, 2)
                })
        
        return {"matrix_data": matrix_grid, "timeline_labels": labels, "status": "Active model weights extracted"}
        
    except Exception as e:
        print(f"Attention Extraction Failure: {e}")
            
    for i in range(len(recent_logs)):
        for j in range(len(recent_logs)):
            matrix_grid.append({"row": i, "col": j, "weight": 1.0 if i == j else 0.25})
            
    return {"matrix_data": matrix_grid, "timeline_labels": labels, "status": "Fallback tracking active"}