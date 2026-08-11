"""
Senate AI - The Parliament Runtime
Orchestrates the Router, Senators, Grouper, Challenger, and voting.
"""

import torch
import yaml
import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import random


class Senate:
    """The full Senate AI parliament system"""
    
    def __init__(self, config_path='config.yaml'):
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
        with open('senate_bundles/senate_index.json') as f:
            self.index = json.load(f)
        
        self.session_history = []
        print(f"Senate ready: {len(self.index.get('senators', []))} senators")
    
    def router(self, question):
        """Simple keyword-based router"""
        question_lower = question.lower()
        
        topic_keywords = {
            'mathematics': ['math', 'calculate', 'number', 'equation', 'formula', 'prime', 'derivative'],
            'physics': ['physics', 'force', 'energy', 'motion', 'gravity', 'light', 'speed', 'mass'],
            'chemistry': ['chemistry', 'chemical', 'element', 'reaction', 'molecule', 'atom', 'bond'],
            'biology': ['biology', 'cell', 'dna', 'organism', 'species', 'evolution', 'gene'],
            'computer_science': ['computer', 'code', 'algorithm', 'program', 'software', 'data', 'binary'],
            'history': ['history', 'war', 'ancient', 'century', 'revolution', 'empire', 'civilization'],
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
            return ['logic', 'philosophy']
        
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
    
    def get_senator_answer(self, senator_info, question):
        """Generate a senator's answer based on specialties"""
        specialties = senator_info['specialties']
        primary = specialties[0].replace('_', ' ')
        
        answers = {
            'mathematics': f"From a mathematical perspective, this involves analyzing patterns and applying logical reasoning.",
            'physics': f"Based on physics principles, this relates to fundamental forces and energy interactions.",
            'biology': f"Biologically speaking, this connects to living systems and cellular processes.",
            'computer_science': f"From a computing standpoint, this can be approached algorithmically.",
            'history': f"Historically, similar patterns have emerged throughout human civilization.",
            'philosophy': f"Philosophically, this raises deeper questions about knowledge and existence.",
            'logic': f"Logically analyzing this, we can break it down into clear premises and conclusions.",
            'psychology': f"From a psychological view, this involves cognitive processes and behavior patterns.",
            'economics': f"Economically, this relates to resource allocation and market dynamics.",
            'linguistics': f"Linguistically, this involves patterns of language and communication.",
            'astronomy': f"From an astronomical perspective, this relates to celestial phenomena.",
            'medicine': f"Medically, this involves understanding biological systems and treatments.",
            'law': f"Legally, this involves principles of justice and established precedents.",
            'art_history': f"From an art historical view, this reflects cultural expression and creativity.",
            'music_theory': f"Musically, this involves patterns of sound and harmonic relationships.",
            'environmental_science': f"Environmentally, this relates to ecosystem dynamics and sustainability.",
            'engineering': f"From an engineering perspective, this involves systematic design and problem-solving.",
        }
        
        return answers.get(primary, f"Based on my expertise in {primary}, I can analyze this question from multiple angles.")
    
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
                if similarity > 0.4:
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
            "Are there unstated assumptions in this conclusion?",
            "Does this answer cover edge cases and exceptions?",
            "Is there empirical evidence supporting this claim?",
            "Could there be alternative explanations worth considering?",
            "Is the reasoning chain logically complete?",
        ]
        return f"CHALLENGE: {random.choice(challenges)} The answer may need revision."
    
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
        """Public interface: Ask the Senate a question"""
        print(f"\n{'='*60}")
        print(f"  SENATE DEBATE")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}")
        
        # Route
        print("\nRouter: Identifying relevant topics...")
        relevant_topics = self.router(question)
        print(f"Topics: {', '.join(relevant_topics)}")
        
        # Select senators
        print("\nSelecting senators...")
        selected = self.select_senators(relevant_topics)
        print(f"{len(selected)} senators selected")
        for s in selected:
            print(f"  Senator {s['senator_id']}: {', '.join(s['specialties'][:3])}")
        
        # Round 1: Independent answers
        print(f"\n{'─'*60}")
        print("  ROUND 1 - Independent Answers")
        print(f"{'─'*60}")
        
        answers = []
        for senator_info in selected:
            answer = self.get_senator_answer(senator_info, question)
            answers.append((senator_info['senator_id'], answer))
        
        # Group and vote
        groups = self.grouper(answers)
        for i, g in enumerate(groups):
            print(f"  Group {i+1}: {g['count']} votes")
        
        consensus, confidence = self.vote(groups)
        
        # Round 2: Challenge
        print(f"\n{'─'*60}")
        print("  ROUND 2 - Challenge & Reconsider")
        print(f"{'─'*60}")
        
        challenge = self.challenger_review(consensus, question)
        print(f"  {challenge}")
        
        # Second round answers
        answers2 = []
        for senator_info in selected:
            answer = self.get_senator_answer(senator_info, f"{question} (Consider: {challenge})")
            answers2.append((senator_info['senator_id'], answer))
        
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
        "Why does ice float on water?",
        "What is the best way to learn programming?",
        "How do we know if an argument is valid?",
    ]
    
    for q in questions:
        senate.ask(q)
        print("\n" + "="*60)
