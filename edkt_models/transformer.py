import torch
import torch.nn as nn

class EDKTTransformer(nn.Module):
    def __init__(self, num_skills, embed_dim=128, num_heads=8, num_layers=4):
        super(EDKTTransformer, self).__init__()
        
        # 1. Embeddings: Converts raw IDs into vectors the AI can understand
        self.skill_embed = nn.Embedding(num_skills + 1, embed_dim)
        self.interaction_embed = nn.Embedding(num_skills * 2 + 1, embed_dim)
        
        # 2. The Transformer Encoder: This is the "Self-Attention" part
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Layer: Predicts the probability of correctness (0.0 to 1.0)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skill_ids, interaction_ids):
        # Combine skill and interaction embeddings
        s_emb = self.skill_embed(skill_ids)
        i_emb = self.interaction_embed(interaction_ids)
        
        x = s_emb + i_emb # Element-wise addition
        
        # Pass through the "Attention" layers
        output = self.transformer(x)
        
        # Final prediction
        return self.sigmoid(self.fc(output))