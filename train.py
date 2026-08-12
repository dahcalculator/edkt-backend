import torch
import torch.optim as optim
import torch.nn as nn
from database import SessionLocal
import models as db_models
from models.transformer import EDKTTransformer

def train_model():
    db = SessionLocal()
    # 1. Load the "Experience" from your research data
    logs = db.query(db_models.InteractionLog).all()
    
    if len(logs) < 5:
        print("Not enough data to train. Go take the quiz a few more times!")
        return

    # 2. Setup the Brain
    model = EDKTTransformer(num_skills=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for Correct/Incorrect

    # 3. Simple Training Loop
    model.train()
    for epoch in range(10): # 10 passes through the data
        optimizer.zero_grad()
        
        # Convert logs to Tensors (simplified for now)
        # In a full research setup, you'd use a DataLoader here
        skills = torch.LongTensor([[log.question_id for log in logs]])
        labels = torch.FloatTensor([[log.is_correct for log in logs]])
        
        # Forward pass: What does the AI think?
        predictions = model(skills, skills) # Simplified interaction mapping
        
        # Calculate Error
        loss = criterion(predictions.squeeze(), labels.squeeze())
        
        # Backward pass: Fix the mistake
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # 4. Save the "Smart" Brain
    torch.save(model.state_dict(), "edkt_model.pth")
    print("AI Training Complete. Model saved as edkt_model.pth")

if __name__ == "__main__":
    train_model()