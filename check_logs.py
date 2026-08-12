from database import SessionLocal
import models

db = SessionLocal()
logs = db.query(models.InteractionLog).all()

print(f"{'ID':<5} | {'User':<5} | {'Correct':<8} | {'Time (sec)':<10}")
print("-" * 40)

for log in logs:
    print(f"{log.id:<5} | {log.user_id:<5} | {log.is_correct:<8} | {log.response_time:<10.2f}")

db.close()