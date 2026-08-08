"""
Senate AI - The Parliament Runtime
Orchestrates the Router, Senators, Grouper, Challenger, and voting.
"""

import torch
import yaml
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from model_template import SenateBundle, Senator
import re
from difflib import SequenceMatcher


class Senate:
    """
    The full Senate AI parliament system.
    Routes questions, runs debates, and reaches consensus.
    """
    
    def __init__(self, config_path='config.yaml', models_dir='models'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(models_dir)
        self.senate_config = self.config['senate']
        
        # Load senate index
        with open(self.models_dir / 'senate_index.json') as f:
            self.index = json.load(f)
        
        # Cache for loaded bundles
        self.loaded_bundles = {}
        self.active_senators = {}
        
        # Performance tracking
        self.session_history = []
        
        print(f"🏛️  Senate AI initialized")
        print(f"   {self.index['total_senators']} senators available")
        print(f"   {len(self.index['topics'])} specialties covered")
    
    def router(self, question):
        """
        Router: Determine which topics are relevant to the question.
        Returns list of relevant specialty topics.
        """
        question_lower = question.lower()
        
        # Keyword matching to find relevant topics
        topic_scores = defaultdict(float)
        
        # Topic keyword mappings
        topic_keywords = {
            'mathematics': ['math', 'calculate', 'number', 'equation', 'formula', 'prime', 'derivative', 'integral'],
            'physics': ['physics', 'force', 'energy', 'motion', 'gravity', 'light', 'speed', 'mass'],
            'chemistry': ['chemistry', 'chemical', 'element', 'reaction', 'molecule', 'atom', 'bond'],
            'biology': ['biology', 'cell', 'dna', 'organism', 'species', 'evolution', 'gene'],
            'computer_science': ['computer', 'code', 'algorithm', 'program', 'software', 'data', 'binary'],
            'history': ['history', 'war', 'ancient', 'century', 'revolution', 'empire', 'civilization'],
            'philosophy': ['philosophy', 'ethic', 'moral', 'existence', 'meaning', 'consciousness', 'reality'],
            'logic': ['logic', 'reason', 'argument', 'valid', 'fallacy', 'premise', 'conclusion'],
            'psychology': ['psychology', 'mind', 'behavior', 'cognitive', 'emotion', 'mental', 'brain'],
            'economics': ['economy', 'market', 'money', 'trade', 'supply', 'demand', 'price'],
            'linguistics': ['language', 'grammar', 'word', 'syntax', 'meaning', 'semantic', 'phonetic'],
            'astronomy': ['space', 'star', 'planet', 'galaxy', 'universe', 'cosmic', 'orbit'],
            'medicine': ['medicine', 'disease', 'treatment', 'symptom', 'diagnosis', 'drug', 'surgery'],
            'law': ['law', 'legal', 'right', 'constitution', 'court', 'justice', 'crime'],
            'art_history': ['art', 'painting', 'sculpture', 'artist', 'renaissance', 'modern', 'aesthetic'],
            'music_theory': ['music', 'note', 'chord', 'rhythm', 'melody', 'harmony', 'scale'],
            'environmental_science': ['environment', 'climate', 'ecosystem', 'pollution', 'sustainable', 'carbon'],
            'engineering': ['engineer', 'design', 'build', 'structure', 'machine', 'system', 'technical'],
        }
        
        # Score each topic based on keyword matches
        for topic, keywords in topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            
            # Also check if topic name itself appears
            if topic in question_lower or topic.replace('_', ' ') in question_lower:
                score += 3
            
            if score > 0:
                topic_scores[topic] = score
        
        # If no matches, include general topics
        if not topic_scores:
            return ['logic', 'philosophy', 'general_knowledge']
        
        # Return top matching topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, score in sorted_topics[:6]]
        
        return top_topics
    
    def select_senators(self, relevant_topics, max_senators=8):
        """
        Select senators whose specialties match the relevant topics.
        """
        senator_scores = []
        
        for senator_info in self.index['senators']:
            senator_topics = set(senator_info['specialties'])
            relevant_set = set(relevant_topics)
            
            # Calculate overlap
            overlap = senator_topics & relevant_set
            
            if overlap:
                # Score based on overlap size
                score = len(overlap) / len(relevant_set)
                
                # Bonus for exact specialty matches
                bonus = sum(
                    1 for topic in overlap 
                    if topic in senator_topics
                )
                score += bonus * 0.1
                
                senator_scores.append((senator_info, score))
        
        # Sort by relevance score
        senator_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top senators (ensure diversity)
        selected = []
        covered_topics = set()
        
        for senator_info, score in senator_scores:
            if len(selected) >= max_senators:
                break
            
            # Check if this senator adds new topic coverage
            new_topics = set(senator_info['specialties']) - covered_topics
            
            if new_topics or len(selected) < 3:
                selected.append(senator_info)
                covered_topics.update(senator_info['specialties'])
        
        return selected[:max_senators]
    
    def load_senator(self, senator_info):
        """Load a senator from its bundle"""
        senator_id = senator_info['senator_id']
        
        # Check if already loaded
        if senator_id in self.active_senators:
            return self.active_senators[senator_id]
        
        bundle_id = senator_info['bundle_id']
        
        # Load bundle if not cached
        if bundle_id not in self.loaded_bundles:
            bundle_path = self.models_dir / senator_info['bundle_file']
            self.loaded_bundles[bundle_id] = SenateBundle.load(bundle_path)
        
        # Get senator from bundle
        bundle = self.loaded_bundles[bundle_id]
        senator = bundle.get_senator(senator_id)
        
        # Cache in active senators
        self.active_senators[senator_id] = senator
        
        return senator
    
    def get_senator_answer(self, senator, question):
        """
        Get a senator's independent answer.
        In production, this uses proper tokenization and generation.
        """
        # Simplified answer generation based on specialties
        specialties_str = ', '.join(senator.specialties)
        
        # Template-based generation (would be replaced with actual model inference)
        answers = {
            'mathematics': "This can be solved mathematically by applying the relevant formulas and principles.",
            'physics': "From a physics perspective, this involves fundamental forces and energy considerations.",
            'biology': "Biologically, this relates to cellular processes and organism functions.",
            'computer_science': "Computationally, this can be approached algorithmically.",
            'history': "Historically, this has precedents in past events and developments.",
            'philosophy': "Philosophically, this raises questions about knowledge and existence.",
            'logic': "Logically, we can analyze this through structured reasoning.",
            'psychology': "Psychologically, this involves mental processes and behavior patterns.",
        }
        
        # Generate answer based on primary specialty
        primary_specialty = senator.specialties[0]
        base_answer = answers.get(primary_specialty, f"Based on my expertise in {specialties_str}, I can analyze this question.")
        
        return base_answer
    
    def grouper(self, answers):
        """
        Grouper: Group similar answers together for voting.
        """
        groups = []
        used = set()
        
        for i, (senator_id, answer) in enumerate(answers):
            if i in used:
                continue
            
            group = {
                'senators': [senator_id],
                'answer': answer,
                'count': 1
            }
            
            # Find similar answers
            for j, (other_id, other_answer) in enumerate(answers):
                if j <= i or j in used:
                    continue
                
                similarity = SequenceMatcher(None, answer.lower(), other_answer.lower()).ratio()
                
                if similarity > 0.6:  # Similarity threshold
                    group['senators'].append(other_id)
                    group['count'] += 1
                    used.add(j)
            
            groups.append(group)
            used.add(i)
        
        # Sort by group size
        groups.sort(key=lambda x: x['count'], reverse=True)
        
        return groups
    
    def challenger_review(self, consensus, question):
        """
        Challenger: Critically review the current consensus.
        """
        challenges = [
            "Are there any unstated assumptions in this conclusion?",
            "Does this answer cover edge cases?",
            "Is there empirical evidence supporting this?",
            "Could there be alternative explanations?",
            "Is the reasoning logically sound?",
        ]
        
        # Select a challenge based on the question
        import random
        challenge = random.choice(challenges)
        
        return f"CHALLENGE: {challenge} The current answer may need revision to address this concern."
    
    def vote(self, groups):
        """
        Conduct voting on answer groups.
        Returns the winning answer and confidence score.
        """
        if not groups:
            return None, 0.0
        
        total_votes = sum(group['count'] for group in groups)
        
        if total_votes == 0:
            return None, 0.0
        
        # Get the leading group
        leading_group = groups[0]
        consensus = leading_group['answer']
        confidence = leading_group['count'] / total_votes
        
        return consensus, confidence
    
    def debate(self, question, max_rounds=None):
        """
        Run a full Senate debate on a question.
        """
        if max_rounds is None:
            max_rounds = self.senate_config['max_rounds']
        
        print(f"\n{'='*60}")
        print(f"  SENATE DEBATE")
        print(f"{'='*60}")
        print(f"\n❓ Question: {question}")
        
        # Phase 1: Route to relevant topics
        print("\n🔍 Router: Identifying relevant topics...")
        relevant_topics = self.router(question)
        print(f"   Topics: {', '.join(relevant_topics)}")
        
        # Phase 2: Select senators
        print("\n👥 Selecting senators...")
        selected_senators = self.select_senators(relevant_topics)
        print(f"   {len(selected_senators)} senators selected")
        
        # Load senators
        senators = []
        for senator_info in selected_senators:
            senator = self.load_senator(senator_info)
            senators.append(senator)
            print(f"   Senator {senator.model_id}: {', '.join(senator.specialties)}
