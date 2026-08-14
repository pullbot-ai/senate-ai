"""
Senate AI - Topic-Based Training with Grading
Trains all senators matching given topics across all bundles.
Grades them on Q&A pairs and updates reliability scores.
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
from difflib import SequenceMatcher


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


def tokenize_text(text, max_len=32, vocab_size=8000):
    """Tokenize text for senator inference"""
    words = text.lower().split()[:max_len]
    tokens = []
    for w in words:
        tokens.append(hash(w) % vocab_size)
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.tensor([tokens])


def decode_tokens(token_ids, vocab):
    """Decode token IDs back to text"""
    words = []
    for tid in token_ids:
        tid = tid.item()
        if tid == 0:
            break
        if tid == 1:
            words.append('?')
        elif tid == 2:
            break
        elif tid in vocab:
            words.append(vocab[tid])
    return ' '.join(words)


def build_vocab():
    """Build reverse vocabulary from training data"""
    vocab = {0: '<PAD>', 1: '<UNK>', 2: '<END>'}
    training_dir = Path('training_data')
    if training_dir.exists():
        for file in training_dir.glob('*.json'):
            try:
                with open(file) as f:
                    data = json.load(f)
                examples = data.get('training_examples', [])
                for text in examples:
                    for word in text.lower().split():
                        tid = hash(word) % 8000
                        if tid not in vocab and tid >= 3:
                            vocab[tid] = word
            except:
                pass
    return vocab


def grade_senator(senator, qa_pairs, vocab):
    """Grade senator on Q&A pairs and return scores"""
    senator.eval()
    scores = []
    
    for qa in qa_pairs:
        question = qa.get('question', '')
        correct_answer = qa.get('answer', '')
        topic = qa.get('topic', '')
        
        # Get senator's answer
        input_ids = tokenize_text(question)
        
        with torch.no_grad():
            logits = senator(input_ids)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits / 0.8, dim=-1)
            
            # Generate 10 tokens
            generated = []
            current = input_ids
            for _ in range(10):
                logits = senator(current)
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == 2:
                    break
                generated.append(next_token)
                current = torch.cat([current, torch.tensor([[next_token]])], dim=1)
        
        senator_answer = decode_tokens(torch.tensor(generated), vocab)
        
        # Score answer
        similarity = SequenceMatcher(None, correct_answer.lower(), senator_answer.lower()).ratio()
        keyword_overlap = set(correct_answer.lower().split()) & set(senator_answer.lower().split())
        keyword_score = min(1.0, len(keyword_overlap) / max(1, len(correct_answer.split()) * 0.3))
        
        total_score = (similarity * 0.6 + keyword_score * 0.4) * 100
        total_score = min(100, max(0, total_score))
        
        scores.append({
            'topic': topic,
            'question': question,
            'correct_answer': correct_answer,
            'senator_answer': senator_answer,
            'score': total_score
        })
    
    return scores


def grade_and_update(senator, topics, vocab):
    """Grade senator on Q&A pairs and update performance"""
    
    all_qa = []
    for topic in topics:
        _, qa_pairs = get_topic_data(topic)
        for qa in qa_pairs:
            qa['topic'] = topic
        all_qa.extend(qa_pairs)
    
    if not all_qa:
        return None
    
    # Sample up to 10 Q&A pairs
    sample_qa = random.sample(all_qa, min(10, len(all_qa)))
    
    scores = grade_senator(senator, sample_qa, vocab)
    
    if not scores:
        return None
    
    # Update performance per topic
    topic_scores = {}
    for s in scores:
        topic = s['topic']
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(s['score'])
    
    for topic, topic_score_list in topic_scores.items():
        avg_score = sum(topic_score_list) / len(topic_score_list)
        senator.performance[topic] = avg_score / 100
    
    avg_score = sum(s['score'] for s in scores) / len(scores)
    return {
        'average_score': avg_score,
        'scores': scores
    }


def train_topics(topic_list, epochs=3, lr=0.001):
    """Train all senators matching topics, grade them, update reliability"""
    
    with open('senate_bundles/senate_index.json') as f:
        index = json.load(f)
    
    topics = set(topic_list)
    print(f"\n{'='*60}")
    print(f"  TOPIC TRAINING WITH GRADING")
    print(f"{'='*60}")
    print(f"  Topics: {', '.join(sorted(topics))}")
    sys.stdout.flush()
    
    vocab = build_vocab()
    
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
    graded = 0
    total_score = 0
    
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
            
            # Train
            losses = train_senator_on_topics(senator, relevant, epochs=epochs, lr=lr)
            
            if losses:
                trained += 1
                bundle_changed = True
                print(f"train done", end=' ')
            else:
                print(f"no data", end=' ')
                skipped += 1
                continue
            
            # Grade
            grading_result = grade_and_update(senator, relevant, vocab)
            
            if grading_result:
                score = grading_result['average_score']
                total_score += score
                graded += 1
                print(f"| grade: {score:.1f}/100")
            else:
                print(f"| no Q&A data")
            
            sys.stdout.flush()
        
        if bundle_changed:
            size_mb = bundle.save(bundle_path)
            print(f"    Saved ({size_mb:.1f}MB)")
            sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trained: {trained} senators")
    print(f"  Graded: {graded} senators")
    if graded > 0:
        print(f"  Avg score: {total_score/graded:.1f}/100")
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
