"""
Senate AI - The Parliament Runtime
Real senator inference with trained models.
"""

import torch
import yaml
import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import random
import sys


class Senate:
    """The full Senate AI parliament system"""
    
    def __init__(self):
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
        with open('senate_bundles/senate_index.json') as f:
            self.index = json.load(f)
        
        # Cache for loaded bundles
        self.loaded_bundles = {}
        self.active_senators = {}
        
        self.session_history = []
        print(f"Senate ready: {len(self.index.get('senators', []))} senators")
    
    def _load_senator(self, senator_info):
        """Actually load a senator model from its bundle"""
        senator_id = senator_info['senator_id']
        
        if senator_id in self.active_senators:
            return self.active_senators[senator_id]
        
        bundle_id = senator_info['bundle_id']
        
        if bundle_id not in self.loaded_bundles:
            bundle_path = f"senate_bundles/bundle_{bundle_id:03d}.pt"
            self.loaded_bundles[bundle_id] = torch.load(bundle_path, map_location='cpu', weights_only=False)
        
        bundle = self.loaded_bundles[bundle_id]
        senators = bundle.get('senators', {})
        
        if str(senator_id) in senators:
            data = senators[str(senator_id)]
        elif senator_id in senators:
            data = senators[senator_id]
        else:
            return None
        
        # Rebuild senator from saved state
        from model_template import Senator
        config = data.get('config', {})
        state_dict = data.get('state_dict', data)
        
        senator = Senator(
            model_id=config.get('model_id', senator_id),
            specialties=config.get('specialties', ['general'])
        )
        
        # Clean state dict (remove any non-tensor stuff)
        clean_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                clean_state[k] = v
        
        senator.load_state_dict(clean_state, strict=False)
        senator.eval()
        
        self.active_senators[senator_id] = senator
        return senator
    
    def _tokenize(self, text, max_len=32):
        """Simple word-based tokenizer"""
        words = text.lower().split()[:max_len]
        tokens = []
        for w in words:
            tokens.append(hash(w) % 8000)
        while len(tokens) < max_len:
            tokens.append(0)
        return torch.tensor([tokens])
    
    def _decode(self, token_ids):
        """Decode token IDs back to text"""
        # Simple reverse lookup using common words
        common_words = ['the', 'a', 'is', 'of', 'in', 'to', 'and', 'that', 'it', 'for',
                       'this', 'with', 'on', 'are', 'be', 'as', 'at', 'from', 'or', 'an',
                       'by', 'not', 'but', 'have', 'has', 'was', 'were', 'they', 'their', 'we',
                       'can', 'all', 'will', 'would', 'could', 'should', 'may', 'also', 'some', 'its']
        
        words = []
        for tid in token_ids:
            tid = tid.item()
            if tid == 0 or tid == 2:
                break
            if 3 <= tid < 3 + len(common_words):
                words.append(common_words[tid - 3])
            elif tid == 1:
                words.append('?')
        
        return ' '.join(words) if words else "..."
    
    def _senator_inference(self, senator, question):
        """Run actual inference on a senator model"""
        input_ids = self._tokenize(question)
        
        with torch.no_grad():
            logits = senator(input_ids)
            
            # Get last token predictions
            last_logits = logits[0, -1, :]
            
            # Top-k sampling
            top_k = 10
            top_values, top_indices = torch.topk(last_logits, top_k)
            probs = torch.softmax(top_values, dim=-1)
            
            # Generate 15-25 tokens
            generated = []
            current = input_ids
            
            for _ in range(random.randint(15, 25)):
                logits = senator(current)
                last_logits = logits[0, -1, :]
                
                top_values, top_indices = torch.topk(last_logits, min(top_k, len(last_logits)))
                probs = torch.softmax(top_values * 0.8, dim=-1)
                
                next_token = top_indices[torch.multinomial(probs, 1)].item()
                
                if next_token == 2:
                    break
                
                generated.append(next_token)
                current = torch.cat([current, torch.tensor([[next_token]])], dim=1)
                
                if len(generated) >= 30:
                    break
            
            return self._decode(torch.tensor(generated))
    
    def router(self, question):
        """Keyword-based router"""
        question_lower = question.lower()
        
        topic_keywords = {
            'mathematics': ['math', 'calculate', 'number', 'equation', 'formula', 'prime', 'derivative'],
            'physics': ['physics', 'force', 'energy', 'motion', 'gravity', 'light', 'speed', 'mass'],
            'chemistry': ['chemistry', 'chemical', 'element', 'reaction', 'molecule', 'atom', 'bond'],
            'biology': ['biology', 'cell', 'dna', 'organism', 'species', 'evolution', 'gene'],
            'computer_science': ['computer', 'code', 'algorithm', 'program', 'software', 'data', 'binary'],
            'history': ['history', 'war', 'ancient', 'century', 'revolution', 'empire', 'civilization', 'king', 'queen', 'president', 'founded'],
            'geography': ['capital', 'country', 'city', 'france', 'paris', 'london', 'continent', 'ocean', 'river', 'mountain', 'border', 'europe', 'asia'],
            'philosophy': ['philosophy', 'ethic', 'moral', 'existence', 'meaning', 'consciousness'],
            'logic': ['logic', 'reason', 'argument', 'valid', 'fallacy', 'premise', 'conclusion'],
            'psychology': ['psychology', 'mind', 'behavior', 'cognitive', 'emotion', 'mental'],
            'economics': ['economy', 'market', 'money', 'trade', 'supply', 'demand', 'price'],
            'linguistics': ['language', 'grammar', 'word', 'syntax', 'meaning', 'semantic'],
            'astronomy': ['space', 'star', 'planet', 'galaxy', 'universe', 'cosmic', 'orbit'],
            'medicine': ['medicine', 'disease', 'treatment', 'symptom', 'diagnosis', 'drug'],
            'law': ['law', 'legal', 'right', 'constitution', 'court', 'justice', 'crime'],
            'art_history': ['art', 'painting', 'sculpture', 'artist', 'renaissance', 'modern'],
            'music_theory': ['music', 'note', 'chord', 'rhythm', 'melody', 'harmony', 'scale'],
            'environmental_science': ['environment', 'climate', 'ecosystem', 'pollution', 'sustainable'],
            'engineering': ['engineer', 'design', 'build', 'structure', 'machine', 'technical'],
        }
        
        topic_scores = defaultdict(float)
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    topic_scores[topic] += 1
            if topic.replace('_', ' ') in question_lower:
                topic_scores[topic] += 3
        
        if not topic_scores:
            return ['logic', 'philosophy', 'history', 'geography']
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:6]]
    
    def select_senators(self, relevant_topics, max_senators=8):
        """Select senators matching relevant topics"""
        senator_scores = []
        
        for senator_info in self.index['senators']:
            senator_topics = set(senator_info['specialties'])
            relevant_set = set(relevant_topics)
            overlap = senator_topics & relevant_set
            
            if overlap:
                score = len(overlap) / len(relevant_set)
                bonus = sum(1 for t in overlap if t in senator_topics) * 0.1
                senator_scores.append((senator_info, score + bonus))
        
        senator_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        covered_topics = set()
        for senator_info, score in senator_scores:
            if len(selected) >= max_senators:
                break
            new_topics = set(senator_info['specialties']) - covered_topics
            if new_topics or len(selected) < 3:
                selected.append(senator_info)
                covered_topics.update(senator_info['specialties'])
        
        return selected[:max_senators]
    
    def grouper(self, answers):
        """Group similar answers together"""
        groups = []
        used = set()
        
        for i, (senator_id, answer) in enumerate(answers):
            if i in used:
                continue
            
            group = {'senators': [senator_id], 'answer': answer, 'count': 1}
            
            for j, (other_id, other_answer) in enumerate(answers):
                if j <= i or j in used:
                    continue
                similarity = SequenceMatcher(None, answer.lower(), other_answer.lower()).ratio()
                if similarity > 0.3:
                    group['senators'].append(other_id)
                    group['count'] += 1
                    used.add(j)
            
            groups.append(group)
            used.add(i)
        
        groups.sort(key=lambda x: x['count'], reverse=True)
        return groups
    
    def challenger_review(self, consensus, question):
        """Challenge the current consensus"""
        challenges = [
            "Are there unstated assumptions?",
            "Does this cover edge cases?",
            "Is there evidence for this?",
            "Could there be alternative explanations?",
            "Is the reasoning complete?",
        ]
        return f"CHALLENGE: {random.choice(challenges)}"
    
    def vote(self, groups):
        """Vote on answer groups"""
        if not groups:
            return "No consensus reached.", 0.0
        
        total_votes = sum(g['count'] for g in groups)
        if total_votes == 0:
            return "No consensus reached.", 0.0
        
        leading = groups[0]
        return leading['answer'], leading['count'] / total_votes
    
    def ask(self, question):
        """Public interface: Ask the Senate a question with real inference"""
        print(f"\n{'='*60}")
        print(f"  SENATE DEBATE")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}")
        sys.stdout.flush()
        
        # Route
        print("\nRouter: Identifying relevant topics...")
        relevant_topics = self.router(question)
        print(f"Topics: {', '.join(relevant_topics)}")
        sys.stdout.flush()
        
        # Select senators
        print("\nSelecting senators...")
        selected = self.select_senators(relevant_topics)
        print(f"{len(selected)} senators selected")
        for s in selected:
            print(f"  Senator {s['senator_id']}: {', '.join(s['specialties'][:3])}")
        sys.stdout.flush()
        
        # Round 1: Real inference
        print(f"\n{'─'*60}")
        print("  ROUND 1 - Independent Answers")
        print(f"{'─'*60}")
        sys.stdout.flush()
        
        answers = []
        for senator_info in selected:
            senator = self._load_senator(senator_info)
            if senator is None:
                continue
            
            print(f"  Senator {senator_info['senator_id']} thinking...", end=' ')
            sys.stdout.flush()
            
            answer = self._senator_inference(senator, question)
            answers.append((senator_info['senator_id'], answer))
            print(f'"{answer[:60]}..."')
            sys.stdout.flush()
        
        # Group and vote
        groups = self.grouper(answers)
        print(f"\n  Groups formed: {len(groups)}")
        for i, g in enumerate(groups[:5]):
            print(f"  Group {i+1}: {g['count']} votes - \"{g['answer'][:50]}...\"")
        sys.stdout.flush()
        
        consensus, confidence = self.vote(groups)
        
        # Round 2: Challenge & reconsider
        print(f"\n{'─'*60}")
        print("  ROUND 2 - Challenge & Reconsider")
        print(f"{'─'*60}")
        sys.stdout.flush()
        
        challenge = self.challenger_review(consensus, question)
        print(f"  {challenge}")
        sys.stdout.flush()
        
        answers2 = []
        for senator_info in selected:
            senator = self._load_senator(senator_info)
            if senator is None:
                continue
            
            reconsider_prompt = f"{question} Reconsider: {challenge}"
            print(f"  Senator {senator_info['senator_id']} reconsidering...", end=' ')
            sys.stdout.flush()
            
            answer = self._senator_inference(senator, reconsider_prompt)
            answers2.append((senator_info['senator_id'], answer))
            print(f'"{answer[:60]}..."')
            sys.stdout.flush()
        
        groups2 = self.grouper(answers2)
        consensus2, confidence2 = self.vote(groups2)
        
        if confidence2 >= confidence:
            consensus = consensus2
            confidence = confidence2
            rounds = 2
        else:
            rounds = 1
        
        print(f"\n{'='*60}")
        print(f"  FINAL ANSWER")
        print(f"{'='*60}")
        print(f"\n{consensus}")
        print(f"\nConfidence: {confidence:.1%}")
        print(f"Rounds: {rounds}")
        print(f"Senators involved: {len(selected)}")
        print(f"Topics: {', '.join(relevant_topics)}")
        sys.stdout.flush()
        
        result = {
            'question': question,
            'consensus': consensus,
            'confidence': confidence,
            'rounds': rounds,
            'senators_involved': len(selected),
            'topics': relevant_topics
        }
        
        self.session_history.append(result)
        return result


if __name__ == "__main__":
    senate = Senate()
    
    questions = [
        "What is the capital of France?",
        "Why does ice float?",
        "How do computers work?",
    ]
    
    for q in questions:
        senate.ask(q)
        print("\n" + "="*60)
