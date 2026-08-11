"""
Senate AI - Training Data Generator
Uses GitHub Models (free GPT-4o) to generate rich training examples.
"""

import json
import os
import requests
import time
from pathlib import Path
import yaml

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
API_URL = "https://models.inference.ai.azure.com/chat/completions"

def call_ai(prompt, max_tokens=300):
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
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            
            print(f"   API error {r.status_code}: {r.text[:100]}")
            time.sleep(5)
        
        except Exception as e:
            print(f"   Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    
    return None


def generate_training_examples(topic, num_examples=20):
    """Generate training examples for a topic using AI"""
    
    prompt = f"""Generate {num_examples} training examples for a tiny AI model specializing in '{topic}'.
Each example should be a single sentence teaching a key concept.
Make them diverse: definitions, principles, applications, examples, and facts.
Return as a JSON array of strings. Format: ["example1", "example2", ...]"""
    
    response = call_ai(prompt, max_tokens=500)
    
    if response:
        try:
            # Extract JSON array from response
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples[:num_examples]
        except:
            pass
    
    # Fallback: basic examples
    return [
        f"{topic} involves understanding key principles and concepts",
        f"A fundamental aspect of {topic} is pattern recognition",
        f"Experts in {topic} apply specialized knowledge to solve problems",
    ]


def generate_qa_pairs(topic, num_pairs=15):
    """Generate Q&A pairs for grading using AI"""
    
    prompt = f"""Generate {num_pairs} question-answer pairs for testing knowledge of '{topic}'.
Each pair should test understanding of a key concept.
Questions should be clear and specific.
Answers should be 1-2 sentences, accurate and concise.
Return as a JSON array of objects with 'question' and 'answer' keys.
Format: [{{"question": "...", "answer": "..."}}, ...]"""
    
    response = call_ai(prompt, max_tokens=800)
    
    if response:
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                pairs = json.loads(match.group())
                return pairs[:num_pairs]
        except:
            pass
    
    # Fallback
    return [
        {"question": f"What is {topic}?", "answer": f"{topic} is the study of fundamental principles and their applications."}
    ]


def generate_all_data():
    """Generate training data for all topics"""
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    print("Generating training data with GitHub Models (GPT-4o)...")
    print("=" * 60)
    
    total_examples = 0
    total_qa = 0
    
    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] {topic}...")
        
        # Generate training examples
        print("   Generating examples...")
        examples = generate_training_examples(topic)
        
        # Generate Q&A pairs
        print("   Generating Q&A pairs...")
        qa_pairs = generate_qa_pairs(topic)
        
        # Save topic data
        topic_data = {
            'topic': topic,
            'training_examples': examples,
            'qa_pairs': qa_pairs,
            'example_count': len(examples),
            'qa_count': len(qa_pairs)
        }
        
        with open(data_dir / f"{topic}.json", 'w') as f:
            json.dump(topic_data, f, indent=2)
        
        total_examples += len(examples)
        total_qa += len(qa_pairs)
        
        print(f"   ✅ {len(examples)} examples, {len(qa_pairs)} Q&A pairs")
        
        # Rate limit
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {total_examples} examples and {total_qa} Q&A pairs")
    print(f"   Across {len(topics)} topics")
    print(f"   Saved to {data_dir}/")


if __name__ == "__main__":
    generate_all_data()
