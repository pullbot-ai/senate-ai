"""
Senate AI - Initialize the Senate
Creates 89 bundles of 45 senators each.
Every senator gets a unique combination of 4-5 specialty topics.
All start from the same template, diverge through training.
"""

import torch
import yaml
import random
import json
import numpy as np
from pathlib import Path
from model_template import Senator, SenateBundle, SENATORS_PER_BUNDLE, TOTAL_BUNDLES, TOTAL_SENATORS
from collections import Counter


def load_config():
    """Load configuration"""
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def assign_specialties(topics, min_spec=4, max_spec=5):
    """Assign random unique specialties to a senator"""
    num_specs = random.randint(min_spec, max_spec)
    return sorted(random.sample(topics, num_specs))


def validate_distribution(senator_configs, topics):
    """Ensure all topics are covered across senators"""
    all_specialties = []
    for config in senator_configs:
        all_specialties.extend(config['specialties'])
    
    topic_counts = Counter(all_specialties)
    
    print("\nTopic Distribution:")
    print(f"Total specialty assignments: {len(all_specialties)}")
    print(f"Unique topics covered: {len(topic_counts)}")
    print(f"Average per topic: {len(all_specialties) / len(topics):.1f}")
    
    uncovered = set(topics) - set(topic_counts.keys())
    if uncovered:
        print(f"\n⚠️  Uncovered topics: {uncovered}")
    
    # Print top and bottom coverage
    print("\nMost covered topics:")
    for topic, count in topic_counts.most_common(5):
        print(f"  {topic}: {count} senators")
    
    print("\nLeast covered topics:")
    for topic, count in topic_counts.most_common()[-5:]:
        print(f"  {topic}: {count} senators")
    
    return topic_counts


def create_senate():
    """Create the entire Senate of 4,005 specialized models"""
    
    config = load_config()
    topics = config['topics']
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  SENATE AI - Initializing the Senate")
    print("=" * 60)
    print(f"\nCreating {TOTAL_SENATORS} senators across {TOTAL_BUNDLES} bundles")
    print(f"Each senator specializes in {config['specialties_per_model']['min']}-{config['specialties_per_model']['max']} topics")
    print(f"Available topics: {len(topics)}")
    
    # Generate all senator configurations
    all_senator_configs = []
    
    for senator_id in range(TOTAL_SENATORS):
        specialties = assign_specialties(
            topics,
            config['specialties_per_model']['min'],
            config['specialties_per_model']['max']
        )
        
        all_senator_configs.append({
            'model_id': senator_id,
            'specialties': specialties,
            'vocab_size': config['training']['vocab_size'],
            'embed_dim': config['training']['embed_dim'],
            'hidden_dim': config['training']['hidden_dim']
        })
    
    # Validate topic distribution
    topic_counts = validate_distribution(all_senator_configs, topics)
    
    # Create bundles
    print("\n" + "=" * 60)
    print("  Creating bundles...")
    print("=" * 60)
    
    senate_index = []
    
    for bundle_id in range(TOTAL_BUNDLES):
        start_idx = bundle_id * SENATORS_PER_BUNDLE
        end_idx = start_idx + SENATORS_PER_BUNDLE
        bundle_configs = all_senator_configs[start_idx:end_idx]
        
        # Create bundle
        bundle = SenateBundle(bundle_id, bundle_configs)
        
        # Save bundle
        bundle_path = models_dir / f"bundle_{bundle_id:03d}.pt"
        size_mb = bundle.save(bundle_path)
        
        # Add to index
        for config in bundle_configs:
            senate_index.append({
                'senator_id': config['model_id'],
                'specialties': config['specialties'],
                'bundle_id': bundle_id,
                'bundle_file': f"bundle_{bundle_id:03d}.pt"
            })
        
        print(f"Bundle {bundle_id:03d}: senators {start_idx}-{end_idx-1} | {size_mb:.1f}MB")
    
    # Save master index
    index = {
        'name': 'Senate AI',
        'total_senators': TOTAL_SENATORS,
        'total_bundles': TOTAL_BUNDLES,
        'senators_per_bundle': SENATORS_PER_BUNDLE,
        'topics': topics,
        'topic_distribution': dict(topic_counts),
        'senators': senate_index,
        'created': str(Path.cwd())
    }
    
    index_path = models_dir / "senate_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  Senate initialized successfully!")
    print("=" * 60)
    print(f"\n📁 Models directory: {models_dir.absolute()}")
    print(f"📋 Index file: {index_path}")
    print(f"👥 Total senators: {TOTAL_SENATORS}")
    print(f"📦 Total bundles: {TOTAL_BUNDLES}")
    
    # Calculate total size
    total_size = sum(
        (models_dir / f"bundle_{i:03d}.pt").stat().st_size 
        for i in range(TOTAL_BUNDLES)
    )
    print(f"💾 Total size: {total_size / (1024*1024):.0f}MB")
    
    return senate_index


if __name__ == "__main__":
    create_senate()
