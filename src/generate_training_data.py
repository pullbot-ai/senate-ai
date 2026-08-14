"""
Senate AI - Training Data Generator
Uses Puter.js free AI to generate rich training examples.
"""

import json
import time
from pathlib import Path
import yaml
from ai_client import call_ai


def generate_training_examples(topic, num_examples=20):
    """Generate training examples for a topic using AI"""
    
    prompt = f"""Generate {num_examples} training examples for a tiny AI model specializing in '{topic}'.
Each example should be a single sentence teaching a key concept.
Make them diverse: definitions, principles, applications, examples, and facts.
Return as a JSON array of strings. Format: ["example1", "example2", ...]"""
    
    response = call_ai(prompt, max_tokens=500)
    
    if response:
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples[:num_examples]
        except:
            pass
    
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
    
    print("Generating training data with Puter.js AI...")
    print("=" * 60)
    
    total_examples = 0
    total_qa = 0
    
    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] {topic}...")
        
        print("   Generating examples...")
        examples = generate_training_examples(topic)
        
        print("   Generating Q&A pairs...")
        qa_pairs = generate_qa_pairs(topic)
        
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
        
        print(f"   {len(examples)} examples, {len(qa_pairs)} Q&A pairs")
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"Generated {total_examples} examples and {total_qa} Q&A pairs")
    print(f"Across {len(topics)} topics")
    print(f"Saved to {data_dir}/")


if __name__ == "__main__":
    generate_all_data()
