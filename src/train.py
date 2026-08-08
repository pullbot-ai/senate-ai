"""
Senate AI - Train a Senate Bundle
Trains all 45 senators in one bundle on their specific topics.
Each senator only trains on its specialty topics, creating diversity.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
import sys
from pathlib import Path
from model_template import SenateBundle
import random


class TopicDataset(Dataset):
    """Training data specific to a topic"""
    
    def __init__(self, topics, texts, seq_length=64, vocab_size=8000):
        self.topics = topics
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Build vocabulary from texts
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<END>': 2}
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        # Cap vocabulary size
        self.word_to_idx = dict(list(self.word_to_idx.items())[:vocab_size])
        
        # Convert texts to token sequences
        self.sequences = []
        for text in texts:
            tokens = [self.word_to_idx.get(w, 1) for w in text.lower().split()]
            tokens.append(2)  # Add END token
            
            # Pad or truncate
            if len(tokens) < seq_length:
                tokens += [0] * (seq_length - len(tokens))
            else:
                tokens = tokens[:seq_length]
            
            self.sequences.append(torch.tensor(tokens))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx][:-1]  # Input tokens
        y = self.sequences[idx][1:]   # Target tokens
        return x, y


def get_topic_data(topic):
    """
    Get training data for a specific topic.
    In production, this loads from curated datasets.
    """
    
    # Training examples per topic
    topic_data = {
        'mathematics': [
            "a prime number is only divisible by one and itself",
            "the derivative measures the rate of change of a function",
            "the integral calculates the area under a curve",
            "pythagorean theorem relates sides of right triangles",
            "a matrix is a rectangular array of numbers",
            "probability measures the likelihood of events occurring",
        ],
        'physics': [
            "force equals mass times acceleration",
            "energy cannot be created or destroyed only transformed",
            "the speed of light is constant in all reference frames",
            "every action has an equal and opposite reaction",
            "gravity attracts objects with mass toward each other",
            "entropy of an isolated system always increases",
        ],
        'computer_science': [
            "an algorithm is a step by step procedure for solving problems",
            "a binary search tree organizes data for efficient lookup",
            "recursion occurs when a function calls itself",
            "a hash table provides constant time average case lookup",
            "object oriented programming organizes code into classes",
            "a neural network is inspired by biological neurons",
        ],
        'logic': [
            "if all men are mortal and socrates is a man then socrates is mortal",
            "a valid argument has a conclusion that follows from premises",
            "modus ponens states if p implies q and p is true then q is true",
            "a contradiction cannot be true under any circumstances",
            "deductive reasoning moves from general to specific",
            "inductive reasoning draws general conclusions from specific cases",
        ],
        'history': [
            "the roman empire fell in four hundred seventy six ce",
            "the renaissance began in italy during the fourteenth century",
            "world war one started in nineteen fourteen",
            "the industrial revolution transformed manufacturing",
            "the french revolution began in seventeen eighty nine",
            "ancient egypt built pyramids as tombs for pharaohs",
        ],
        'philosophy': [
            "plato believed in a world of ideal forms",
            "aristotle emphasized empirical observation",
            "descartes declared i think therefore i am",
            "kant argued that morality requires universal principles",
            "existentialism emphasizes individual freedom and choice",
            "utilitarianism seeks the greatest good for the greatest number",
        ],
        'biology': [
            "cells are the basic unit of life",
            "dna contains the genetic instructions for organisms",
            "natural selection drives evolution of species",
            "mitosis produces two identical daughter cells",
            "photosynthesis converts sunlight into chemical energy",
            "enzymes catalyze biochemical reactions in cells",
        ],
        'psychology': [
            "classical conditioning associates stimuli with responses",
            "cognitive dissonance occurs when beliefs conflict with actions",
            "the unconscious mind influences behavior according to freud",
            "maslow hierarchy of needs starts with physiological requirements",
            "operant conditioning uses rewards and punishments",
            "confirmation bias leads people to favor supporting evidence",
        ],
    }
    
    if topic in topic_data:
        return topic_data[topic]
    
    # Generate generic training data for topics without specific data
    return [
        f"{topic} involves understanding key principles and concepts",
        f"studying {topic} requires analytical thinking and reasoning",
        f"experts in {topic} apply specialized knowledge to problems",
        f"the field of {topic} continues to evolve with new research",
        f"fundamental concepts in {topic} build on established theories",
        f"{topic} connects to many other areas of knowledge",
    ]


def train_senator(senator, specialties, epochs=3, lr=0.001, batch_size=8):
    """Train a single senator on its specialty topics"""
    
    # Collect training data from all senator specialties
    all_texts = []
    for specialty in specialties:
        texts = get_topic_data(specialty)
        all_texts.extend(texts)
    
    # Shuffle for better training
    random.shuffle(all_texts)
    
    # Create dataset and dataloader
    dataset = TopicDataset(specialties, all_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.Adam(senator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    senator.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = senator(batch_x)
            
            # Calculate loss
            loss = criterion(logits.permute(0, 2, 1), batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 1 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return losses


def train_bundle(bundle_id):
    """Train all 45 senators in a bundle"""
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load the bundle
    bundle_path = f"models/bundle_{bundle_id:03d}.pt"
    
    if not Path(bundle_path).exists():
        print(f"❌ Bundle {bundle_id} not found. Run initialize first.")
        return
    
    print(f"📦 Loading bundle {bundle_id}...")
    bundle = SenateBundle.load(bundle_path)
    
    print(f"\n{'='*60}")
    print(f"  Training Senate Bundle {bundle_id}")
    print(f"  Senators: {len(bundle.senators)}")
    print(f"{'='*60}")
    
    # Train each senator
    for senator_id, senator in bundle.senators.items():
        print(f"\n👤 Senator {senator_id}")
        print(f"   Specialties: {', '.join(senator.specialties)}")
        
        # Train on senator's specialties
        train_senator(
            senator,
            senator.specialties,
            epochs=config['training']['epochs'],
            lr=config['training']['learning_rate'],
            batch_size=config['training']['batch_size']
        )
        
        # Show parameter divergence (how much it changed from template)
        total_change = sum(
            (p != 0).float().sum().item() 
            for p in senator.parameters()
        )
        print(f"   Active parameters: {total_change}")
    
    # Save trained bundle
    size_mb = bundle.save(bundle_path)
    print(f"\n✅ Bundle {bundle_id} trained and saved ({size_mb:.1f}MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <bundle_id>")
        print("Example: python train.py 0")
        sys.exit(1)
    
    bundle_id = int(sys.argv[1])
    train_bundle(bundle_id)
