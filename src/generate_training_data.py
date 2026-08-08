"""
Senate AI - Training Data Generator
Uses AI to generate diverse training examples for each topic.
Run this before training to build rich topic datasets.
"""

import json
from pathlib import Path

# We'll use a simple template system + AI generation patterns
# In production, this could call an API or use a larger model

def generate_topic_data(topic, num_examples=50):
    """
    Generate training examples for a specific topic.
    Uses templates and patterns to create diverse examples.
    """
    
    examples = []
    
    # Base templates for different types of knowledge
    templates = {
        'definition': [
            "{topic} is the study of {aspect}",
            "{topic} refers to {aspect}",
            "The field of {topic} encompasses {aspect}",
            "{topic} can be defined as {aspect}",
        ],
        'principle': [
            "A key principle in {topic} is that {aspect}",
            "In {topic}, it is established that {aspect}",
            "The fundamental concept of {topic} involves {aspect}",
            "According to {topic}, {aspect}",
        ],
        'application': [
            "{topic} can be applied to {aspect}",
            "One application of {topic} is {aspect}",
            "In practice, {topic} helps us {aspect}",
            "Using {topic}, we can {aspect}",
        ],
        'relationship': [
            "{topic} is closely related to {aspect}",
            "There is a connection between {topic} and {aspect}",
            "{topic} and {aspect} share common principles",
            "Understanding {topic} requires knowledge of {aspect}",
        ],
        'method': [
            "To solve problems in {topic}, one must {aspect}",
            "The methodology of {topic} involves {aspect}",
            "A common approach in {topic} is to {aspect}",
            "Experts in {topic} typically {aspect}",
        ],
        'example': [
            "For example, in {topic}, {aspect}",
            "A classic example of {topic} is {aspect}",
            "Consider how {topic} explains {aspect}",
            "An illustration of {topic} can be seen in {aspect}",
        ],
        'comparison': [
            "Unlike other fields, {topic} emphasizes {aspect}",
            "While related, {topic} differs from others in {aspect}",
            "What makes {topic} unique is {aspect}",
            "The distinctive feature of {topic} is {aspect}",
        ],
        'history': [
            "Historically, {topic} developed from {aspect}",
            "The origins of {topic} can be traced to {aspect}",
            "Over time, {topic} has evolved to include {aspect}",
            "Key figures in {topic} contributed {aspect}",
        ],
    }
    
    # Topic-specific aspects to make examples relevant
    topic_aspects = {
        'mathematics': [
            "numbers and their properties",
            "patterns and structures",
            "quantitative relationships",
            "abstract reasoning and logic",
            "mathematical proofs and theorems",
            "algebraic manipulation",
            "geometric visualization",
            "statistical analysis",
            "calculating probabilities",
            "solving equations systematically",
        ],
        'physics': [
            "matter and energy interactions",
            "fundamental forces of nature",
            "motion and mechanics",
            "electromagnetic phenomena",
            "thermodynamic processes",
            "quantum behavior",
            "wave properties",
            "conservation laws",
            "field theories",
            "experimental measurement",
        ],
        'computer_science': [
            "algorithms and data structures",
            "computational thinking",
            "software design patterns",
            "information processing",
            "automated reasoning",
            "digital systems",
            "programming paradigms",
            "complexity analysis",
            "machine learning techniques",
            "network protocols",
        ],
        'logic': [
            "valid reasoning patterns",
            "deductive arguments",
            "premise-conclusion relationships",
            "logical fallacies",
            "truth tables",
            "syllogistic reasoning",
            "propositional logic",
            "predicate calculus",
            "modal logic",
            "informal reasoning",
        ],
        'philosophy': [
            "questions about existence",
            "nature of knowledge",
            "ethical frameworks",
            "consciousness and mind",
            "free will and determinism",
            "aesthetics and beauty",
            "political philosophy",
            "metaphysical concepts",
            "epistemological inquiry",
            "phenomenological experience",
        ],
        'biology': [
            "living organisms",
            "cellular processes",
            "genetic inheritance",
            "evolutionary adaptation",
            "ecosystem dynamics",
            "molecular mechanisms",
            "physiological functions",
            "developmental biology",
            "biodiversity patterns",
            "biochemical pathways",
        ],
        'psychology': [
            "human behavior patterns",
            "cognitive processes",
            "emotional responses",
            "social interactions",
            "developmental stages",
            "personality traits",
            "mental health",
            "learning mechanisms",
            "memory formation",
            "perception and attention",
        ],
        'history': [
            "past civilizations",
            "cultural developments",
            "political transformations",
            "economic systems",
            "social movements",
            "technological innovations",
            "military conflicts",
            "intellectual revolutions",
            "demographic changes",
            "institutional evolution",
        ],
        'economics': [
            "market mechanisms",
            "supply and demand",
            "resource allocation",
            "monetary policy",
            "fiscal systems",
            "international trade",
            "economic growth",
            "labor markets",
            "financial instruments",
            "behavioral incentives",
        ],
        'linguistics': [
            "language structures",
            "grammatical rules",
            "semantic meaning",
            "phonetic patterns",
            "syntactic analysis",
            "language acquisition",
            "dialectal variation",
            "pragmatic usage",
            "etymological origins",
            "communication theory",
        ],
    }
    
    # Default aspects for topics not in the dictionary
    default_aspects = [
        f"the fundamental principles of {topic}",
        f"key concepts in {topic}",
        f"important theories in {topic}",
        f"practical applications of {topic}",
        f"research methods in {topic}",
        f"historical development of {topic}",
        f"modern understanding of {topic}",
        f"critical analysis of {topic}",
        f"emerging trends in {topic}",
        f"interdisciplinary connections to {topic}",
    ]
    
    aspects = topic_aspects.get(topic, default_aspects)
    
    # Generate examples by combining templates with aspects
    import random
    random.seed(hash(topic) % 10000)  # Reproducible but varied by topic
    
    for i in range(num_examples):
        # Select random template type and template
        template_type = random.choice(list(templates.keys()))
        template = random.choice(templates[template_type])
        
        # Select random aspect
        aspect = random.choice(aspects)
        
        # Format the example
        example = template.format(topic=topic, aspect=aspect)
        
        # Add some variation
        if random.random() < 0.3:
            # Add a follow-up sentence
            follow_ups = [
                f" This is essential for understanding the broader field.",
                f" Many researchers have explored this area extensively.",
                f" Students often find this concept particularly interesting.",
                f" This principle has numerous real-world applications.",
                f" Understanding this leads to deeper insights.",
            ]
            example += random.choice(follow_ups)
        
        examples.append(example)
    
    return examples


def generate_all_training_data():
    """Generate training data for all topics in config"""
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    all_data = {}
    
    print("Generating training data for all topics...")
    print("="*60)
    
    for topic in topics:
        examples = generate_topic_data(topic, num_examples=50)
        all_data[topic] = examples
        
        # Save individual topic file
        topic_file = data_dir / f"{topic}.json"
        with open(topic_file, 'w') as f:
            json.dump({
                'topic': topic,
                'examples': examples,
                'count': len(examples)
            }, f, indent=2)
        
        print(f"✅ {topic}: {len(examples)} examples")
    
    # Save combined dataset
    combined_file = data_dir / "all_topics.json"
    with open(combined_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    total_examples = sum(len(v) for v in all_data.values())
    
    print(f"\n{'='*60}")
    print(f"Generated {total_examples} total examples across {len(topics)} topics")
    print(f"Data saved to {data_dir}/")
    
    return all_data


if __name__ == "__main__":
    generate_all_training_data()
