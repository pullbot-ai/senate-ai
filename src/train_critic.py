"""
Senate AI - Critic Trainer
Trains the Challenger to distinguish good answers from bad ones.
AI generates examples with increasing difficulty levels.
"""

import os
import json
import yaml
import requests
import time
import random
from pathlib import Path

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
API_URL = "https://models.inference.ai.azure.com/chat/completions"


def call_ai(prompt, max_tokens=1000):
    """Call GitHub Models API"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(3):
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
                timeout=60
            )
            
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            
            print(f"   API error {r.status_code}: {r.text[:100]}")
            time.sleep(5)
        except Exception as e:
            print(f"   Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    
    return None


def generate_critic_examples(topic, difficulty, num_examples=10):
    """
    Generate examples for training the critic.
    Returns pairs of (answer, is_good, feedback_if_bad)
    
    Difficulty levels:
    1 - Obvious errors (beginner)
    2 - Subtle mistakes (intermediate)
    3 - Edge cases and nuance (advanced)
    4 - Expert-level reasoning (expert)
    5 - Near-impossible distinctions (master)
    """
    
    difficulty_descriptions = {
        1: "Make the bad answers have obvious factual errors, logical fallacies, or completely miss the point.",
        2: "Make the bad answers partially correct but with subtle flaws in reasoning or missing key details.",
        3: "Make the bad answers seem correct at first glance but contain nuanced errors or overlook edge cases.",
        4: "Make the bad answers expertly written with only a tiny flaw that requires deep knowledge to spot.",
        5: "Make both answers highly sophisticated. The bad one should be wrong in a way only a true expert would catch."
    }
    
    prompt = f"""Generate {num_examples} question-answer pairs for training an AI critic.

Topic: {topic}
Difficulty: {difficulty}/5 - {difficulty_descriptions.get(difficulty, '')}

For each question, provide TWO answers:
1. A "good" answer that is accurate and well-reasoned
2. A "flawed" answer with specific problems

For the flawed answer, include specific feedback explaining:
- What exactly is wrong
- Why it's wrong
- How to fix it

Return as JSON array:
[
  {{
    "question": "...",
    "good_answer": "...",
    "flawed_answer": "...",
    "flaw_type": "factual_error|logical_fallacy|incomplete|misleading|oversimplified",
    "feedback": "Specific explanation of what's wrong and how to fix it",
    "difficulty": {difficulty}
  }},
  ...
]"""
    
    response = call_ai(prompt, max_tokens=2000)
    
    if response:
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples
        except Exception as e:
            print(f"   Parse error: {e}")
    
    return []


def generate_questions_only(topic, difficulty, num_questions=10):
    """Generate questions for the critic to evaluate"""
    
    prompt = f"""Generate {num_questions} challenging questions about '{topic}'.
Difficulty level: {difficulty}/5

These questions will be used to test an AI's critical thinking.
Mix of: factual, conceptual, edge cases, and reasoning questions.

Return as JSON array of strings: ["question1", "question2", ...]"""
    
    response = call_ai(prompt, max_tokens=800)
    
    if response:
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
    
    return [f"Explain the key concepts of {topic}."]


def generate_all_critic_data():
    """Generate training data for all topics across all difficulty levels"""
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    
    output_dir = Path('critic_training')
    output_dir.mkdir(exist_ok=True)
    
    all_examples = []
    all_questions = []
    
    # Generate for each difficulty level
    for difficulty in range(1, 6):
        print(f"\n{'='*60}")
        print(f"  DIFFICULTY LEVEL {difficulty}/5")
        print(f"{'='*60}")
        
        level_examples = []
        level_questions = []
        
        # Sample topics (not all 55, to save time)
        sample_topics = random.sample(topics, min(15, len(topics)))
        
        for i, topic in enumerate(sample_topics):
            print(f"\n[{i+1}/{len(sample_topics)}] {topic} (difficulty {difficulty})...")
            
            # Generate critic training examples
            print("   Generating good/bad answer pairs...")
            examples = generate_critic_examples(topic, difficulty)
            
            if examples:
                level_examples.extend(examples)
                all_examples.extend(examples)
                print(f"   ✅ {len(examples)} pairs generated")
            else:
                print(f"   ⚠️  Failed to generate")
            
            # Generate test questions
            print("   Generating test questions...")
            questions = generate_questions_only(topic, difficulty)
            
            if questions:
                for q in questions:
                    level_questions.append({
                        "question": q,
                        "topic": topic,
                        "difficulty": difficulty
                    })
                all_questions.extend(level_questions[-len(questions):])
                print(f"   ✅ {len(questions)} questions")
            
            time.sleep(1)
        
        # Save per-level data
        level_data = {
            "difficulty": difficulty,
            "examples": level_examples,
            "questions": level_questions,
            "example_count": len(level_examples),
            "question_count": len(level_questions)
        }
        
        with open(output_dir / f"difficulty_{difficulty}.json", 'w') as f:
            json.dump(level_data, f, indent=2)
        
        print(f"\n   Level {difficulty} saved: {len(level_examples)} examples, {len(level_questions)} questions")
    
    # Save combined data
    combined = {
        "total_examples": len(all_examples),
        "total_questions": len(all_questions),
        "by_difficulty": {},
        "examples": all_examples,
        "questions": all_questions
    }
    
    # Organize by difficulty
    for ex in all_examples:
        diff = ex.get('difficulty', 1)
        if str(diff) not in combined['by_difficulty']:
            combined['by_difficulty'][str(diff)] = []
        combined['by_difficulty'][str(diff)].append(ex)
    
    with open(output_dir / "all_critic_data.json", 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  CRITIC TRAINING DATA COMPLETE")
    print(f"{'='*60}")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Difficulty levels: 1-5")
    print(f"  Saved to: {output_dir}/")
    
    # Print difficulty breakdown
    for diff in range(1, 6):
        count = len(combined['by_difficulty'].get(str(diff), []))
        print(f"    Level {diff}: {count} examples")


if __name__ == "__main__":
    generate_all_critic_data()
