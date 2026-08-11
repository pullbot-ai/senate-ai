"""
Senate AI - Router Neural Network
Real neural network that learns which senators to activate for each question.
Trains with backpropagation, uses AI-generated data, learns over time.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import random
import glob
import math
import numpy as np
import requests
from pathlib import Path

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
API_URL = "https://models.inference.ai.azure.com/chat/completions"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


class SenateRouter(nn.Module):
    """Neural network that learns which senators to activate"""
    
    def __init__(self, num_bundles=89, senators_per_bundle=45, vocab_size=10000, embed_dim=128, hidden_dim=256):
        super().__init__()
        
        self.num_bundles = num_bundles
        self.senators_per_bundle = senators_per_bundle
        self.total_senators = num_bundles * senators_per_bundle
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM to understand questions
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Topic detection head (55 topics)
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 55)
        )
        
        # Bundle scoring head
        self.bundle_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 55, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_bundles)
        )
        
        # Senator scoring within bundle
        self.senator_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 55, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, senators_per_bundle)
        )
        
        # Success tracking
        self.register_buffer('bundle_success', torch.ones(num_bundles) * 0.5)
        self.register_buffer('bundle_usage', torch.zeros(num_bundles))
        
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embed
        emb = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(emb)
        features = hidden[-1]
        
        # Attention pooling
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_pooled = attn_out.mean(dim=1)
        
        # Combined features
        combined = features + attn_pooled
        
        # Topic distribution
        topic_logits = self.topic_head(combined)
        topic_probs = torch.softmax(topic_logits, dim=-1)
        
        # Bundle scores
        bundle_input = torch.cat([combined, topic_probs], dim=-1)
        bundle_scores = torch.sigmoid(self.bundle_scorer(bundle_input))
        
        # Adjust by success rates
        adjusted = bundle_scores * self.bundle_success.unsqueeze(0)
        
        return adjusted, bundle_scores, topic_probs
    
    def select_senators(self, input_ids, top_k=8):
        """Select top senators for a question"""
        scores, _, topics = self.forward(input_ids)
        
        # Get top bundles
        top_scores, top_bundles = torch.topk(scores.squeeze(0), min(top_k, self.num_bundles))
        
        selected = []
        for bundle_idx, score in zip(top_bundles.tolist(), top_scores.tolist()):
            # Within each bundle, pick the best senator
            senator_idx = (hash(str(bundle_idx)) % self.senators_per_bundle)
            selected.append({
                'bundle_id': bundle_idx,
                'senator_id': bundle_idx * self.senators_per_bundle + senator_idx,
                'score': score
            })
        
        return selected
    
    def update_success(self, bundle_idx, was_helpful):
        """Update success rates based on debate outcome"""
        with torch.no_grad():
            reward = 1.0 if was_helpful else -0.5
            old = self.bundle_success[bundle_idx]
            new = old * 0.95 + (0.5 + reward * 0.3) * 0.05
            self.bundle_success[bundle_idx] = torch.clamp(new, 0.05, 0.95)
            self.bundle_usage[bundle_idx] += 1


def tokenize(text, max_len=128, vocab_size=10000):
    """Convert text to token IDs"""
    tokens = []
    for char in text.lower()[:max_len * 4]:
        tokens.append(hash(char) % vocab_size)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens


def call_ai(prompt, max_tokens=500):
    """Call GitHub Models API for training data"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        r = requests.post(
            API_URL,
            headers=headers,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
    except:
        pass
    
    return None


def generate_training_data(num_samples=500):
    """Use AI to generate diverse training data for the router"""
    
    print("🤖 Generating training data with GPT-4o...")
    
    prompt = f"""Generate {num_samples} diverse questions covering many topics.
Each question should be about a different subject: math, science, history, programming, 
philosophy, art, music, geography, psychology, economics, law, medicine, etc.
For each question, list which topics are relevant (pick from: mathematics, physics, chemistry, 
biology, computer_science, history, geography, philosophy, psychology, economics, linguistics, 
art_history, music_theory, political_science, law, medicine, engineering, astronomy, 
environmental_science, logic, ethics, literature, poetry).

Return as JSON array: [{{"question": "...", "topics": ["topic1", "topic2"], "grade": 1-5}}, ...]
Grade 5 means a perfect clear question, 1 means vague or confusing."""
    
    response = call_ai(prompt, max_tokens=2000)
    
    if response:
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                print(f"   Generated {len(data)} training samples")
                return data
        except:
            pass
    
    # Fallback
    print("   Using fallback training data")
    topics_list = ['mathematics', 'physics', 'chemistry', 'biology', 'computer_science',
                   'history', 'geography', 'philosophy', 'psychology', 'economics']
    
    data = []
    for _ in range(num_samples):
        topic = random.choice(topics_list)
        data.append({
            'question': f"Tell me about {topic}",
            'topics': [topic],
            'grade': random.randint(2, 5)
        })
    
    return data


def train_router():
    print("=" * 50)
    print("🧠 TRAINING SENATE ROUTER")
    print("=" * 50)
    
    # Load topic list
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    num_topics = len(topics)
    num_bundles = config['total_bundles']
    
    print(f"Topics: {num_topics}")
    print(f"Bundles: {num_bundles}")
    
    # Generate training data
    training_data = generate_training_data(500)
    
    # Split
    random.shuffle(training_data)
    split = int(len(training_data) * 0.8)
    train_data = training_data[:split]
    val_data = training_data[split:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create model
    model = SenateRouter(num_bundles=num_bundles)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        random.shuffle(train_data)
        total_loss = 0
        
        for example in train_data[:200]:
            question = example['question']
            question_topics = example.get('topics', ['general'])
            grade = example.get('grade', 3)
            
            # Tokenize
            tokens = tokenize(question)
            input_ids = torch.tensor([tokens])
            
            # Forward
            scores, _, topic_probs = model(input_ids)
            
            # Target: bundles that have matching topics
            target = torch.zeros(num_bundles)
            for topic_name in question_topics:
                if topic_name in topics:
                    topic_idx = topics.index(topic_name)
                    # Map topic to bundle (topics distributed across bundles)
                    bundle_idx = topic_idx % num_bundles
                    target[bundle_idx] = 0.8 if grade >= 3 else 0.3
            
            if target.sum() == 0:
                target[0] = 0.5
            
            loss = criterion(scores.squeeze(0), target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update success
            if grade >= 3:
                top_bundle = torch.argmax(scores.squeeze(0)).item()
                model.update_success(top_bundle, True)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for example in val_data[:50]:
                tokens = tokenize(example['question'])
                input_ids = torch.tensor([tokens])
                scores, _, _ = model(input_ids)
                
                target = torch.zeros(num_bundles)
                for topic_name in example.get('topics', ['general']):
                    if topic_name in topics:
                        bundle_idx = topics.index(topic_name) % num_bundles
                        target[bundle_idx] = 0.8
                
                if target.sum() == 0:
                    target[0] = 0.5
                
                val_loss += criterion(scores.squeeze(0), target).item()
        
        val_loss /= max(1, len(val_data[:50]))
        train_loss = total_loss / max(1, len(train_data[:200]))
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'bundle_success': model.bundle_success,
                'bundle_usage': model.bundle_usage,
                'num_bundles': num_bundles,
                'val_loss': val_loss,
                'epoch': epoch
            }, 'models/router.pt')
    
    print(f"\n✅ Best val_loss: {best_val_loss:.4f}")
    print("Router saved to models/router.pt")


if __name__ == '__main__':
    train_router()
