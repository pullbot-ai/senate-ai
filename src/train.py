"""
Senate AI - Topic-Based Training
Trains all senators matching given topics across all bundles.
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
    """Training data for a topic"""
    
    def __init__(self, texts, seq_length=64, vocab_size=8000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<END>': 2}
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        self.word_to_idx = dict(list(self.word_to_idx.items())[:vocab_size])
        
        self.sequences = []
        for text in texts:
            tokens = [self.word_to_idx.get(w, 1) for w in text.lower().split()]
            tokens.append(2)
            
            if len(tokens) < seq_length:
                tokens += [0] * (seq_length - len(tokens))
            else:
                tokens = tokens[:seq_length]
            
            self.sequences.append(torch.tensor(tokens))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx][:-1]
        y = self.sequences[idx][1:]
        return x, y


def get_topic_data(topic):
    """Load training data for a topic"""
    data_file = Path(f"training_data/{topic}.json")
    
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
            return data.get('training_examples', []), data.get('qa_pairs', [])
    
    return [], []


def train_senator_on_topics(senator, topics, epochs=3, lr=0.001, batch_size=8):
    """Train a senator on specific topics"""
    
    all_texts = []
    for topic in topics:
        texts, _ = get_topic_data(topic)
        all_texts.extend(texts)
    
    if not all_texts:
        return None
    
    random.shuffle(all_texts)
    dataset = TopicDataset(all_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(senator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    senator.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = senator(batch_x)
            loss = criterion(logits.permute(0, 2, 1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
    
    return losses


def train_topics(topic_list, epochs=3, lr=0.001):
    """Train all senators that specialize in any of the given topics"""
    
    with open('senate_bundles/senate_index.json') as f:
        index = json.load(f)
    
    topics = set(topic_list)
    print(f"\n{'='*60}")
    print(f"  TOPIC TRAINING")
    print(f"{'='*60}")
    print(f"  Topics: {', '.join(sorted(topics))}")
    sys.stdout.flush()
    
    # Find matching senators across all bundles
    matching_senators = []
    for senator in index['senators']:
        senator_topics = set(senator['specialties'])
        if senator_topics & topics:
            matching_senators.append(senator)
    
    print(f"  Matching senators: {len(matching_senators)}")
    sys.stdout.flush()
    
    if not matching_senators:
        print("  No senators match these topics")
        return
    
    # Group by bundle
    bundle_groups = {}
    for senator in matching_senators:
        bid = senator['bundle_id']
        if bid not in bundle_groups:
            bundle_groups[bid] = []
        bundle_groups[bid].append(senator)
    
    print(f"  Across {len(bundle_groups)} bundles")
    sys.stdout.flush()
    
    trained = 0
    skipped = 0
    
    for bundle_id, senators in sorted(bundle_groups.items()):
        bundle_path = f"senate_bundles/bundle_{bundle_id:03d}.pt"
        
        if not Path(bundle_path).exists():
            print(f"  Bundle {bundle_id} not found, skipping {len(senators)} senators")
            skipped += len(senators)
            continue
        
        print(f"\n  Bundle {bundle_id} ({len(senators)} senators)...")
        sys.stdout.flush()
        
        bundle = SenateBundle.load(bundle_path)
        bundle_changed = False
        
        for senator_info in senators:
            senator_id = senator_info['senator_id']
            senator = bundle.get_senator(senator_id)
            
            if senator is None:
                print(f"    Senator {senator_id} not in bundle")
                skipped += 1
                continue
            
            senator_topics = set(senator.specialties)
            relevant = list(senator_topics & topics)
            
            print(f"    Senator {senator_id} [{', '.join(relevant[:3])}]...", end=' ')
            sys.stdout.flush()
            
            losses = train_senator_on_topics(senator, relevant, epochs=epochs, lr=lr)
            
            if losses:
                print(f"loss: {losses[-1]:.4f}")
                trained += 1
                bundle_changed = True
            else:
                print("no data")
                skipped += 1
        
        if bundle_changed:
            size_mb = bundle.save(bundle_path)
            print(f"    Saved ({size_mb:.1f}MB)")
            sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trained: {trained} senators")
    print(f"  Skipped: {skipped}")
    print(f"  Topics: {', '.join(sorted(topics))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', type=str, required=True, help='Comma-separated topics')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    topics = [t.strip() for t in args.topics.split(',')]
    
    train_topics(topics, epochs=args.epochs, lr=args.lr)
