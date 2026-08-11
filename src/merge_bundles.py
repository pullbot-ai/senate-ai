"""
Senate AI - Bundle Merger
Merges multiple topic-trained bundles, keeping the best senator from each.
"""

import torch
import sys
from pathlib import Path
from model_template import SenateBundle


def merge_bundles(bundle_id):
    """Merge topic-trained bundles, keeping best senator versions"""
    
    base_path = Path(f"senate_bundles/bundle_{bundle_id:03d}.pt")
    
    if not base_path.exists():
        print(f"❌ Base bundle {bundle_id} not found")
        return
    
    # Load original bundle
    print(f"Loading base bundle {bundle_id}...")
    base = SenateBundle.load(str(base_path))
    
    # Track best version of each senator
    best_senators = {}
    best_scores = {}
    
    # Score senators from base bundle
    for senator_id, senator in base.senators.items():
        score = senator.get_reliability()
        best_senators[senator_id] = senator
        best_scores[senator_id] = score
    
    print(f"Base: {len(base.senators)} senators, avg score: {sum(best_scores.values())/len(best_scores):.2f}")
    
    # Check for other trained versions
    # They'll be saved with topic prefixes in the same directory
    senate_dir = Path("senate_bundles")
    topic_versions = list(senate_dir.glob(f"bundle_{bundle_id:03d}_*.pt"))
    
    if not topic_versions:
        print("No topic-trained versions found, keeping base")
        base.save(str(base_path))
        return
    
    # Merge each topic version
    for version_path in topic_versions:
        topic_name = version_path.stem.split('_', 2)[-1]
        print(f"\nMerging {topic_name}...")
        
        try:
            topic_bundle = SenateBundle.load(str(version_path))
            improved = 0
            
            for senator_id, senator in topic_bundle.senators.items():
                score = senator.get_reliability()
                
                if senator_id not in best_scores or score > best_scores[senator_id]:
                    best_senators[senator_id] = senator
                    best_scores[senator_id] = score
                    improved += 1
            
            print(f"  Improved: {improved} senators")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Build final bundle with best senators
    final_scores = list(best_scores.values())
    print(f"\n{'='*50}")
    print(f"  MERGE COMPLETE")
    print(f"{'='*50}")
    print(f"  Senators: {len(best_senators)}")
    print(f"  Avg score: {sum(final_scores)/len(final_scores):.3f}")
    print(f"  Best: {max(final_scores):.3f}")
    print(f"  Worst: {min(final_scores):.3f}")
    
    # Update bundle with best senators
    base.senators = best_senators
    base.save(str(base_path))
    print(f"\n  ✅ Saved merged bundle")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_bundles.py <bundle_id>")
        sys.exit(1)
    
    bundle_id = int(sys.argv[1])
    merge_bundles(bundle_id)
