import torch
import torch.nn as nn
import torch.optim as optim
from database import SessionLocal
import models as db_models
from edkt_models.transformer import EDKTTransformer

def train():
    db = SessionLocal()
    
    # 1. Gather all users to map their chronological history timelines
    users = db.query(db_models.User).all()
    
    skills_list = []
    interactions_list = []
    labels_list = []
    
    NUM_SKILLS = 50 # Total vocabulary definition size matching system settings
    
    for user in users:
        # Pull distinct interactions for this specific student, sorted by timestamp sequence
        user_logs = db.query(db_models.InteractionLog).filter(
            db_models.InteractionLog.user_id == user.id
        ).order_by(db_models.InteractionLog.id).all()
        
        # A student track sequence requires at least 2 logs to trace cognitive change over time
        if len(user_logs) < 2:
            continue
            
        # Extract features up to the second-to-last item
        user_skills = [log.question_id for log in user_logs[:-1]]
        user_responses = [log.is_correct for log in user_logs[:-1]]
        
        # Labels are shifted by 1 step into the future (predicting the next response)
        user_labels = [float(log.is_correct) for log in user_logs[1:]]
        
        # Calculate interaction codes: Question ID + (Response * 50)
        user_interactions = [
            q_id + (resp * NUM_SKILLS)
            for q_id, resp in zip(user_skills, user_responses)
        ]
        
        skills_list.append(user_skills)
        interactions_list.append(user_interactions)
        labels_list.append(user_labels)

    db.close()

    # 2. Check if the database has gathered enough interaction sequences
    if len(skills_list) == 0:
        print("❌ Insufficient sequenced data. Please take the quiz a few more times on the frontend to generate historical logs.")
        return

    # 3. Handle Sequence Padding for Batch Processing
    # Pads student histories of varying lengths with zeros to match the longest sequence
    max_len = max(len(seq) for seq in skills_list)
    
    padded_skills = [seq + [0] * (max_len - len(seq)) for seq in skills_list]
    padded_interactions = [seq + [0] * (max_len - len(seq)) for seq in sorted(interactions_list, key=len, reverse=True)] # Matching target sequence index space
    padded_interactions = [seq + [0] * (max_len - len(seq)) for seq in interactions_list]
    padded_labels = [seq + [0.0] * (max_len - len(seq)) for seq in labels_list]

    # Convert native Python arrays to PyTorch Tensors
    skills_tensor = torch.LongTensor(padded_skills)
    interactions_tensor = torch.LongTensor(padded_interactions)
    labels_tensor = torch.FloatTensor(padded_labels)

    # 4. Initialize Transformer Network Architecture
    model = EDKTTransformer(num_skills=NUM_SKILLS)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy evaluates 0.0-1.0 probabilities against true hit metrics

    print(f"Starting Training on MAT101 Interaction Data ({len(skills_list)} qualified user traces)...")
    
    model.train()
    for epoch in range(40): # Extended execution loop for steady gradient descent convergence
        optimizer.zero_grad()
        
        # Pass both distinct sequence features into the self-attentive layers
        output = model(skills_tensor, interactions_tensor)
        
        # Format shapes to strip away trailing single dimensions
        loss = criterion(output.squeeze(-1), labels_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:02d}/40], Cross-Entropy Loss: {loss.item():.4f}")

    # 5. Export optimized weight files for system inference
    torch.save(model.state_dict(), "trained_edkt_model.pth")
    print("Optimization Complete. Model saved as 'trained_edkt_model.pth'.")

if __name__ == "__main__":
    train()