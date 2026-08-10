"""
Senate AI - Initialize the Senate
Creates 89 bundle files (~45MB each) containing all 4,005 senators.
Commits directly to repo.
"""

import torch
import yaml
import random
import json
import numpy as np
from pathlib import Path
from model_template import Senator, SenateBundle
from collections import Counter


def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def assign_specialties(topics, min_spec=4, max_spec=5):
    num_specs = random.randint(min_spec, max_spec)
    return sorted(random.sample(topics, num_specs))


def create_senate():
    config = load_config()
    topics = config['topics']
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    output_dir = Path("senate_bundles")
    output_dir.mkdir(exist_ok=True)
    
    TOTAL_SENATORS = 4005
    SENATORS_PER_BUNDLE = 45
    TOTAL_BUNDLES = 89
    
    print("=" * 60)
    print("  SENATE AI - Initializing the Senate")
    print("=" * 60)
    print(f"\nCreating {TOTAL_SENATORS} senators in {TOTAL_BUNDLES} bundles")
    
    all_configs = []
    for senator_id in range(TOTAL_SENATORS):
        specialties = assign_specialties(topics, 4, 5)
        all_configs.append({
            'model_id': senator_id,
            'specialties': specialties,
            'vocab_size': config['training']['vocab_size'],
            'embed_dim': config['training']['embed_dim'],
            'hidden_dim': config['training']['hidden_dim']
        })
    
    senate_index = []
    
    for bundle_id in range(TOTAL_BUNDLES):
        start = bundle_id * SENATORS_PER_BUNDLE
        end = start + SENATORS_PER_BUNDLE
        bundle_configs = all_configs[start:end]
        
        bundle = SenateBundle(bundle_id, bundle_configs)
        bundle_path = output_dir / f"bundle_{bundle_id:03d}.pt"
        size_mb = bundle.save(bundle_path)
        
        print(f"Bundle {bundle_id:03d}: senators {start}-{end-1} | {size_mb:.1f}MB")
        
        for config in bundle_configs:
            senate_index.append({
                'senator_id': config['model_id'],
                'specialties': config['specialties'],
                'bundle_id': bundle_id,
                'bundle_file': f"bundle_{bundle_id:03d}.pt"
            })
    
    index = {
        'name': 'Senate AI',
        'total_senators': TOTAL_SENATORS,
        'total_bundles': TOTAL_BUNDLES,
        'senators_per_bundle': SENATORS_PER_BUNDLE,
        'topics': topics,
        'senators': senate_index
    }
    
    with open(output_dir / "senate_index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    total_size = sum(
        (output_dir / f"bundle_{i:03d}.pt").stat().st_size 
        for i in range(TOTAL_BUNDLES)
    )
    
    print(f"\n✅ Senate initialized!")
    print(f"📁 {TOTAL_BUNDLES} bundles saved to {output_dir}/")
    print(f"💾 Total size: {total_size / (1024*1024):.0f}MB")
    print(f"👥 {TOTAL_SENATORS} senators ready")
    
    return senate_index


if __name__ == "__main__":
    create_senate()
