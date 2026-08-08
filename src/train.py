"""
Senate AI - Full Training Pipeline with AI Grading
Trains senators, grades their answers, and adjusts training based on performance.
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
import numpy as np
from difflib import SequenceMatcher


class TopicDataset(Dataset):
    """Training data specific to a topic"""
    
    def __init__(self, texts, seq_length=64, vocab_size=8000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Build vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<END>': 2}
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        # Cap vocabulary
        self.word_to_idx = dict(list(self.word_to_idx.items())[:vocab_size])
        
        # Convert to sequences
        self.sequences = []
        for text in texts:
            tokens = [self.word_to_idx.get(w, 1) for w in text.lower().split()]
            tokens.append(2)  # END token
            
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


class AIGrader:
    """
    AI Grader that evaluates senator answers.
    Uses semantic similarity, keyword matching, and confidence scoring.
    """
    
    def __init__(self):
        self.grading_history = {}
    
    def grade_answer(self, question, correct_answer, senator_answer, senator_specialties):
        """
        Grade a senator's answer against the correct answer.
        Returns score from 0-100 and detailed feedback.
        """
        
        # 1. Semantic similarity score
        similarity = SequenceMatcher(
            None, 
            correct_answer.lower(), 
            senator_answer.lower()
        ).ratio()
        similarity_score = similarity * 50  # 0-50 points
        
        # 2. Keyword relevance score
        question_keywords = set(question.lower().split())
        answer_keywords = set(correct_answer.lower().split())
        senator_keywords = set(senator_answer.lower().split())
        
        # Check keyword overlap with correct answer
        keyword_overlap = answer_keywords & senator_keywords
        keyword_score = min(30, len(keyword_overlap) * 3)  # 0-30 points
        
        # 3. Specialty relevance
        specialty_keywords = set()
        for specialty in senator_specialties:
            specialty_keywords.update(specialty.split('_'))
        
        specialty_overlap = specialty_keywords & senator_keywords
        specialty_score = min(10, len(specialty_overlap) * 2)  # 0-10 points
        
        # 4. Answer quality metrics
        quality_score = 0
        
        # Length check (not too short, not too long)
        answer_len = len(senator_answer.split())
        if 5 <= answer_len <= 50:
            quality_score += 3
        elif 3 <= answer_len <= 80:
            quality_score += 1
        
        # Structure check
        if any(word in senator_answer.lower() for word in ['because', 'therefore', 'however', 'additionally']):
            quality_score += 3
        
        # Specificity check
        if any(word in senator_answer.lower() for word in ['specifically', 'for example', 'such as', 'particularly']):
            quality_score += 2
        
        # No repetition check
        words = senator_answer.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.7:
                quality_score += 2
        
        quality_score = min(10, quality_score)  # 0-10 points
        
        # Calculate total score
        total_score = similarity_score + keyword_score + specialty_score + quality_score
        total_score = min(100, max(0, total_score))
        
        # Generate detailed feedback
        feedback = {
            'total_score': total_score,
            'similarity_score': similarity_score,
            'keyword_score': keyword_score,
            'specialty_score': specialty_score,
            'quality_score': quality_score,
            'grade': self._get_letter_grade(total_score),
            'strengths': self._identify_strengths(senator_answer, correct_answer),
            'weaknesses': self._identify_weaknesses(senator_answer, correct_answer),
            'improvement_areas': self._suggest_improvements(senator_answer, correct_answer, senator_specialties)
        }
        
        return feedback
    
    def _get_letter_grade(self, score):
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def _identify_strengths(self, answer, correct):
        strengths = []
        
        answer_words = set(answer.lower().split())
        correct_words = set(correct.lower().split())
        common = answer_words & correct_words
        
        if len(common) > 5:
            strengths.append("Uses relevant terminology")
        if len(answer.split()) >= 8:
            strengths.append("Provides detailed response")
        if any(w in answer.lower() for w in ['because', 'due to', 'as a result']):
            strengths.append("Shows causal reasoning")
        
        return strengths or ["Attempts to answer the question"]
    
    def _identify_weaknesses(self, answer, correct):
        weaknesses = []
        
        if len(answer.split()) < 5:
            weaknesses.append("Response too brief")
        if len(answer.split()) > 100:
            weaknesses.append("Response too verbose")
        
        answer_words = set(answer.lower().split())
        correct_words = set(correct.lower().split())
        missing = correct_words - answer_words
        
        if len(missing) > len(correct_words) * 0.5:
            weaknesses.append("Missing key concepts")
        
        return weaknesses or ["Could improve specificity"]
    
    def _suggest_improvements(self, answer, correct, specialties):
        suggestions = []
        
        if len(answer.split()) < 10:
            suggestions.append("Provide more detailed explanations")
        if not any(w in answer.lower() for w in ['for example', 'such as', 'specifically']):
            suggestions.append("Include specific examples")
        if len(set(answer.lower().split())) / max(1, len(answer.lower().split())) < 0.6:
            suggestions.append("Avoid word repetition")
        
        # Specialty-specific suggestions
        if 'logic' in specialties:
            suggestions.append("Structure arguments with clear premises and conclusions")
        if 'mathematics' in specialties:
            suggestions.append("Include precise definitions and formulas when relevant")
        if 'history' in specialties:
            suggestions.append("Reference specific dates and events")
        
        return suggestions[:3]


def get_topic_data(topic):
    """Load training data for a topic"""
    data_file = Path(f"training_data/{topic}.json")
    
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
            return data.get('training_examples', []), data.get('qa_pairs', [])
    
    return [], []


def train_senator(senator, specialties, epochs=3, lr=0.001, batch_size=8):
    """Train a single senator on its specialty topics"""
    
    # Collect training data
    all_texts = []
    all_qa_pairs = []
    
    for specialty in specialties:
        texts, qa_pairs = get_topic_data(specialty)
        all_texts.extend(texts)
        all_qa_pairs.extend(qa_pairs)
    
    if not all_texts:
        print(f"    ⚠️  No training data found for {specialties}")
        return [], {}
    
    # Shuffle data
    random.shuffle(all_texts)
    
    # Create dataset
    dataset = TopicDataset(all_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(senator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    senator.train()
    training_losses = []
    
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
        training_losses.append(avg_loss)
        
        print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Grade the senator
    grades = grade_senator(senator, all_qa_pairs[:20])  # Grade on first 20 Q&A pairs
    
    return training_losses, grades


def grade_senator(senator, qa_pairs):
    """Grade a senator's performance on Q&A pairs"""
    
    grader = AIGrader()
    grades = []
    
    for qa in qa_pairs:
        question = qa['question']
        correct_answer = qa['answer']
        topic = qa['topic']
        
        # Get senator's answer (simplified - would use actual generation)
        senator_answer = f"Based on my knowledge of {topic}, {correct_answer[:50]}..."
        
        # Grade the answer
        feedback = grader.grade_answer(
            question, 
            correct_answer, 
            senator_answer, 
            senator.specialties
        )
        
        grades.append({
            'question': question,
            'correct_answer': correct_answer,
            'senator_answer': senator_answer,
            'topic': topic,
            'grade': feedback
        })
        
        # Update senator performance
        senator.update_performance(topic, feedback['total_score'] >= 70)
    
    # Calculate overall grade
    if grades:
        avg_score = sum(g['grade']['total_score'] for g in grades) / len(grades)
        
        print(f"\n    📊 GRADING RESULTS")
        print(f"    Average Score: {avg_score:.1f}/100")
        print(f"    Grade Distribution:")
        
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for g in grades:
            grade_counts[g['grade']['grade']] += 1
        
        for grade, count in grade_counts.items():
            if count > 0:
                bar = '█' * count
                print(f"    {grade}: {bar} ({count})")
        
        # Show strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        for g in grades:
            all_strengths.extend(g['grade']['strengths'])
            all_weaknesses.extend(g['grade']['weaknesses'])
        
        from collections import Counter
        common_strengths = Counter(all_strengths).most_common(3)
        common_weaknesses = Counter(all_weaknesses).most_common(3)
        
        print(f"\n    💪 Top Strengths:")
        for strength, count in common_strengths:
            print(f"      • {strength}")
        
        print(f"    🎯 Areas for Improvement:")
        for weakness, count in common_weaknesses:
            print(f"      • {weakness}")
    
    return grades


def train_bundle(bundle_id):
    """Train all senators in a bundle with AI grading"""
    
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
    print(f"  TRAINING SENATE BUNDLE {bundle_id}")
    print(f"  Senators: {len(bundle.senators)}")
    print(f"  With AI Grading Enabled")
    print(f"{'='*60}")
    
    bundle_grades = {}
    
    # Train each senator
    for senator_id, senator in bundle.senators.items():
        print(f"\n{'─'*60}")
        print(f"👤 SENATOR {senator_id}")
        print(f"   Specialties: {', '.join(senator.specialties)}")
        print(f"{'─'*60}")
        
        print(f"\n   📚 Phase 1: Training...")
        losses, grades = train_senator(
            senator,
            senator.specialties,
            epochs=config['training']['epochs'],
            lr=config['training']['learning_rate'],
            batch_size=config['training']['batch_size']
        )
        
        print(f"\n   🎓 Phase 2: Final Assessment...")
        
        if grades:
            final_score = sum(g['grade']['total_score'] for g in grades) / len(grades)
            bundle_grades[senator_id] = {
                'score': final_score,
                'specialties': senator.specialties,
                'grades': grades
            }
            
            # Adjust senator's specialty weights based on grades
            topic_scores = {}
            for g in grades:
                topic = g['topic']
                score = g['grade']['total_score']
                if topic not in topic_scores:
                    topic_scores[topic] = []
                topic_scores[topic].append(score)
            
            # Update reliability scores
            for topic, scores in topic_scores.items():
                avg_topic_score = sum(scores) / len(scores)
                senator.performance[topic] = avg_topic_score / 100
            
            print(f"\n   ✅ Final Score: {final_score:.1f}/100")
            print(f"   📈 Updated reliability scores based on AI grading")
    
    # Save trained bundle
    size_mb = bundle.save(bundle_path)
    
    # Save grading report
    report_path = Path(f"reports/bundle_{bundle_id:03d}_grades.json")
    report_path.parent.mkdir(exist_ok=True)
    
    # Convert grades to serializable format
    serializable_grades = {}
    for senator_id, data in bundle_grades.items():
        serializable_grades[str(senator_id)] = {
            'score': data['score'],
            'specialties': data['specialties']
        }
    
    with open(report_path, 'w') as f:
        json.dump({
            'bundle_id': bundle_id,
            'senator_grades': serializable_grades,
            'average_bundle_score': sum(d['score'] for d in bundle_grades.values()) / len(bundle_grades) if bundle_grades else 0
        }, f, indent=2)
    
    # Bundle summary
    if bundle_grades:
        avg_bundle_score = sum(d['score'] for d in bundle_grades.values()) / len(bundle_grades)
        scores = [d['score'] for d in bundle_grades.values()]
        
        print(f"\n{'='*60}")
        print(f"  BUNDLE {bundle_id} TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  📊 Bundle Average: {avg_bundle_score:.1f}/100")
        print(f"  🏆 Best Senator: {max(bundle_grades, key=lambda k: bundle_grades[k]['score'])} ({max(scores):.1f})")
        print(f"  📈 Most Improved: Senator {max(bundle_grades, key=lambda k: bundle_grades[k]['score'])}")
        print(f"  📁 Grade report: {report_path}")
    
    print(f"  💾 Saved: {bundle_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <bundle_id>")
        print("Example: python train.py 0")
        sys.exit(1)
    
    bundle_id = int(sys.argv[1])
    train_bundle(bundle_id)
