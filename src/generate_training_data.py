"""
Senate AI - Training Data Generator with Q&A pairs
Generates topic data AND question-answer pairs for AI grading.
"""

import json
import random
from pathlib import Path

def generate_qa_pairs(topic, num_pairs=30):
    """Generate question-answer pairs for testing and grading"""
    
    qa_templates = {
        'definition': [
            ("What is {topic}?", "{topic} is the study of {aspect}"),
            ("Define {topic}.", "{topic} can be defined as {aspect}"),
            ("What does {topic} involve?", "The field of {topic} encompasses {aspect}"),
        ],
        'explanation': [
            ("Why is {topic} important?", "{topic} is important because {aspect}"),
            ("How does {topic} work?", "{topic} works through {aspect}"),
            ("What is the purpose of {topic}?", "The purpose of {topic} is to {aspect}"),
        ],
        'application': [
            ("How is {topic} applied?", "{topic} is applied through {aspect}"),
            ("Give an example of {topic}.", "An example of {topic} is {aspect}"),
            ("Where is {topic} used?", "{topic} is used in {aspect}"),
        ],
        'concept': [
            ("What is a key concept in {topic}?", "A key concept in {topic} is {aspect}"),
            ("What principle underlies {topic}?", "The principle underlying {topic} is {aspect}"),
            ("What should I know about {topic}?", "You should know that in {topic}, {aspect}"),
        ],
    }
    
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
    qa_pairs = []
    
    random.seed(hash(topic) % 10000)
    
    for _ in range(num_pairs):
        q_type = random.choice(list(qa_templates.keys()))
        q_template, a_template = random.choice(qa_templates[q_type])
        aspect = random.choice(aspects)
        
        question = q_template.format(topic=topic.replace('_', ' '), aspect=aspect)
        answer = a_template.format(topic=topic.replace('_', ' '), aspect=aspect)
        
        qa_pairs.append({
            'question': question,
            'answer': answer,
            'type': q_type,
            'topic': topic
        })
    
    return qa_pairs


def generate_topic_data(topic, num_examples=50):
    """Generate training examples for a specific topic"""
    
    examples = []
    
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
    
    random.seed(hash(topic) % 10000)
    
    for _ in range(num_examples):
        template_type = random.choice(list(templates.keys()))
        template = random.choice(templates[template_type])
        aspect = random.choice(aspects)
        
        example = template.format(topic=topic.replace('_', ' '), aspect=aspect)
        
        if random.random() < 0.3:
            follow_ups = [
                " This is essential for understanding the broader field.",
                " Many researchers have explored this area extensively.",
                " Students often find this concept particularly interesting.",
                " This principle has numerous real-world applications.",
                " Understanding this leads to deeper insights.",
            ]
            example += random.choice(follow_ups)
        
        examples.append(example)
    
    return examples


def generate_all_data():
    """Generate both training examples and Q&A pairs for all topics"""
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    all_examples = {}
    all_qa = {}
    
    print("Generating training data and Q&A pairs...")
    print("="*60)
    
    for topic in topics:
        # Generate training examples
        examples = generate_topic_data(topic, num_examples=50)
        all_examples[topic] = examples
        
        # Generate Q&A pairs for grading
        qa_pairs = generate_qa_pairs(topic, num_pairs=30)
        all_qa[topic] = qa_pairs
        
        # Save combined topic data
        topic_data = {
            'topic': topic,
            'training_examples': examples,
            'qa_pairs': qa_pairs,
            'example_count': len(examples),
            'qa_count': len(qa_pairs)
        }
        
        topic_file = data_dir / f"{topic}.json"
        with open(topic_file, 'w') as f:
            json.dump(topic_data, f, indent=2)
        
        print(f"✅ {topic}: {len(examples)} examples, {len(qa_pairs)} Q&A pairs")
    
    # Save combined files
    with open(data_dir / "all_examples.json", 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    with open(data_dir / "all_qa_pairs.json", 'w') as f:
        json.dump(all_qa, f, indent=2)
    
    total_examples = sum(len(v) for v in all_examples.values())
    total_qa = sum(len(v) for v in all_qa.values())
    
    print(f"\n{'='*60}")
    print(f"Generated {total_examples} examples and {total_qa} Q&A pairs")
    print(f"Across {len(topics)} topics")
    print(f"Data saved to {data_dir}/")
    
    return all_examples, all_qa


if __name__ == "__main__":
    generate_all_data()
