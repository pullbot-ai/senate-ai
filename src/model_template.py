"""
Senate AI - Model Template
Each senator starts from this template and diverges through training.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path

class Senator(nn.Module):
    """
    ~500K parameter expert model.
    Each senator has unique specialties and develops its own perspective.
    """
    
    def __init__(self, model_id, specialties, vocab_size=8000, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.model_id = model_id
        self.specialties = specialties
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Two-layer LSTM for reasoning
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Personality vector - unique to each senator
        self.personality = nn.Parameter(torch.randn(32))
        
        # Specialty embeddings - how this senator weighs its topics
        self.specialty_weights = nn.Parameter(torch.randn(len(specialties), 16))
        
        # Confidence estimator
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        # Track performance
        self.performance = {specialty: 0.5 for specialty in specialties}
        self.total_votes = 0
        self.correct_votes = 0
        
    def forward(self, x, return_confidence=False):
        batch_size, seq_len = x.shape
        
        # Embed input tokens
        x = self.embed(x)
        
        # Inject personality into representation
        personality_signal = self.personality.unsqueeze(0).unsqueeze(0)
        personality_signal = personality_signal.expand(batch_size, seq_len, -1)
        x = torch.cat([x, personality_signal[:, :, :x.size(-1)]], dim=-1)
        
        # Pad back to embed_dim if needed
        if x.size(-1) != self.embed.embedding_dim:
            x = x[:, :, :self.embed.embedding_dim]
        
        # Process through LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Get final hidden state
        final_state = x[:, -1, :]
        
        # Generate output
        logits = self.output(x)
        
        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(final_state))
            return logits, confidence
        
        return logits
    
    def generate_answer(self, question_tokens, max_length=50):
        """Generate an answer token by token"""
        self.eval()
        
        with torch.no_grad():
            # Start with question tokens
            current = question_tokens.unsqueeze(0)
            generated = []
            
            for _ in range(max_length):
                logits = self.forward(current)
                next_token = logits[0, -1, :].argmax().item()
                generated.append(next_token)
                
                # Append to current sequence
                next_token_tensor = torch.tensor([[next_token]])
                current = torch.cat([current, next_token_tensor], dim=1)
                
                # Stop if end token
                if next_token == 2:  # <END> token
                    break
            
            return generated
    
    def update_performance(self, specialty, was_correct):
        """Update reliability tracking"""
        self.total_votes += 1
        if was_correct:
            self.correct_votes += 1
        
        # Exponential moving average for specialty performance
        alpha = 0.1
        current = self.performance.get(specialty, 0.5)
        self.performance[specialty] = (1 - alpha) * current + alpha * (1.0 if was_correct else 0.0)
    
    def get_reliability(self, specialty=None):
        """Get reliability score"""
        if specialty and specialty in self.performance:
            return self.performance[specialty]
        return self.correct_votes / max(1, self.total_votes)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'model_id': self.model_id,
            'specialties': self.specialties,
            'param_count': sum(p.numel() for p in self.parameters()),
            'performance': self.performance,
            'reliability': self.get_reliability()
        }


class SenateBundle:
    """
    Contains 45 senators in one ~45MB file.
    """
    
    def __init__(self, bundle_id, senator_configs):
        self.bundle_id = bundle_id
        self.senators = {}
        
        for config in senator_configs:
            senator_id = config['model_id']
            self.senators[senator_id] = Senator(**config)
    
    def save(self, path):
        """Save all 45 senators to one file"""
        data = {
            'bundle_id': self.bundle_id,
            'senators': {
                senator_id: {
                    'state_dict': senator.state_dict(),
                    'config': senator.get_config()
                }
                for senator_id, senator in self.senators.items()
            }
        }
        torch.save(data, path)
        
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return size_mb
    
    @classmethod
    def load(cls, path, senator_ids=None):
        """Load specific senators from bundle"""
        data = torch.load(path, map_location='cpu')
        bundle = cls.__new__(cls)
        bundle.bundle_id = data['bundle_id']
        bundle.senators = {}
        
        ids_to_load = senator_ids or data['senators'].keys()
        
        for senator_id in ids_to_load:
            if senator_id in data['senators']:
                senator_data = data['senators'][senator_id]
                config = senator_data['config']
                senator = Senator(
                    model_id=config['model_id'],
                    specialties=config['specialties']
                )
                senator.load_state_dict(senator_data['state_dict'])
                senator.performance = config.get('performance', {})
                bundle.senators[senator_id] = senator
        
        return bundle
    
    def get_senator(self, senator_id):
        return self.senators.get(senator_id)


# Constants
SENATORS_PER_BUNDLE = 45
TOTAL_BUNDLES = 89
TOTAL_SENATORS = 4005
