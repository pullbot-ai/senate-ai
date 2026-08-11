"""
Senate AI - GGUF Exporter
Exports all 4,005 senators into chunked GGUF files (~300MB each).
"""

import torch
import sys
import struct
import numpy as np
from pathlib import Path
from collections import OrderedDict
import json


def write_gguf_file(filepath, tensors_dict, metadata=None):
    """Write multiple senators' tensors into one GGUF file"""
    
    # Flatten all tensors with senator prefix
    all_tensors = OrderedDict()
    total_params = 0
    
    for senator_key, state_dict in tensors_dict.items():
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            full_name = f"{senator_key}.{name}"
            
            # Convert to float16
            if tensor.dtype == torch.float32:
                tensor = tensor.half()
            
            all_tensors[full_name] = tensor
            total_params += tensor.numel()
    
    # Write file
    with open(filepath, 'wb') as f:
        # GGUF magic
        f.write(b'GGUF')
        
        # Version 3
        f.write(struct.pack('<I', 3))
        
        # Number of tensors
        f.write(struct.pack('<Q', len(all_tensors)))
        
        # Metadata count
        if metadata:
            metadata_json = json.dumps(metadata)
            f.write(struct.pack('<Q', 1))
            f.write(struct.pack('<Q', len(b'general.metadata')))
            f.write(b'general.metadata')
            f.write(struct.pack('<I', 8))  # string type
            f.write(struct.pack('<Q', len(metadata_json)))
            f.write(metadata_json.encode('utf-8'))
        else:
            f.write(struct.pack('<Q', 0))
        
        # Calculate offsets
        header_size = f.tell()
        
        # Tensor info size estimation
        tensor_info_size = len(all_tensors) * 128
        data_offset = header_size + tensor_info_size
        data_offset = ((data_offset + 31) // 32) * 32  # Align
        
        # Write tensor infos
        current_offset = data_offset
        for name, tensor in all_tensors.items():
            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Dimensions
            shape = tensor.shape
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<Q', dim))
            
            # Type (1 = F16)
            f.write(struct.pack('<I', 1))
            
            # Offset
            f.write(struct.pack('<Q', current_offset))
            
            # Calculate next offset
            tensor_size = tensor.numel() * 2  # float16 = 2 bytes
            current_offset += tensor_size
            current_offset = ((current_offset + 31) // 32) * 32  # Align
        
        # Write tensor data
        for name, tensor in all_tensors.items():
            # Seek to aligned offset
            f.seek(current_offset - (current_offset - f.tell()) if f.tell() < current_offset else f.tell())
            
            # Pad to alignment
            target = ((f.tell() + 31) // 32) * 32
            while f.tell() < target:
                f.write(b'\x00')
                if f.tell() >= target:
                    break
            
            # Write tensor bytes
            f.write(tensor.numpy().tobytes())
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    return size_mb, total_params


def export_all_bundles(output_name="senate_ai", max_size_mb=300):
    """Export all bundles into chunked GGUF files"""
    
    output_dir = Path("gguf_exports")
    output_dir.mkdir(exist_ok=True)
    
    # Load index
    with open('senate_bundles/senate_index.json') as f:
        index = json.load(f)
    
    total_senators = len(index['senators'])
    print(f"\n{'='*60}")
    print(f"  SENATE AI - GGUF EXPORT")
    print(f"{'='*60}")
    print(f"  Senators: {total_senators}")
    print(f"  Target: {max_size_mb}MB per file")
    print(f"  Format: Float16 (GGUF)")
    print(f"{'='*60}\n")
    
    # Process all bundles
    all_tensors = OrderedDict()
    current_file = 1
    current_size_est = 0
    senators_in_current = 0
    file_info = []
    
    for senator_info in index['senators']:
        senator_id = senator_info['senator_id']
        bundle_id = senator_info['bundle_id']
        
        # Load bundle if not cached
        bundle_path = f"senate_bundles/bundle_{bundle_id:03d}.pt"
        if not hasattr(export_all_bundles, 'bundle_cache'):
            export_all_bundles.bundle_cache = {}
        
        if bundle_id not in export_all_bundles.bundle_cache:
            export_all_bundles.bundle_cache[bundle_id] = torch.load(
                bundle_path, map_location='cpu', weights_only=False
            )
        
        bundle = export_all_bundles.bundle_cache[bundle_id]
        senators = bundle.get('senators', {})
        
        senator_key = str(senator_id)
        if senator_key not in senators:
            continue
        
        data = senators[senator_key]
        state_dict = data.get('state_dict', data)
        
        # Estimate size (params × 2 bytes for float16)
        senator_params = sum(
            v.numel() for v in state_dict.values() 
            if isinstance(v, torch.Tensor)
        )
        senator_size_mb = senator_params * 2 / (1024 * 1024)
        
        # If adding this senator exceeds max size, save current file
        if current_size_est + senator_size_mb > max_size_mb and all_tensors:
            # Save current chunk
            filename = f"{output_name}_{current_file:02d}.gguf"
            filepath = output_dir / filename
            
            metadata = {
                "model": "Senate AI",
                "file": current_file,
                "senators": f"{senators_in_current - len(all_tensors)}-{senators_in_current - 1}" if len(all_tensors) > 0 else "0-0",
                "total_files": "TBD",
                "format": "GGUF F16"
            }
            
            print(f"\n💾 Saving {filename}...")
            size_mb, params = write_gguf_file(filepath, all_tensors, metadata)
            
            file_info.append({
                "file": filename,
                "size_mb": round(size_mb, 1),
                "senators": len(all_tensors),
                "params": params
            })
            
            print(f"   ✅ {size_mb:.1f}MB | {len(all_tensors)} senators | {params:,} params")
            
            # Reset for next file
            all_tensors = OrderedDict()
            current_size_est = 0
            current_file += 1
        
        # Add senator tensors
        senator_prefix = f"s{senator_id:04d}"
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                all_tensors[f"{senator_prefix}.{name}"] = tensor
        
        current_size_est += senator_size_mb
        senators_in_current = senator_id + 1
        
        # Progress
        if senator_id % 100 == 0:
            print(f"\r  Processing senator {senator_id}/{total_senators}...", end='')
            sys.stdout.flush()
    
    # Save final file
    if all_tensors:
        filename = f"{output_name}_{current_file:02d}.gguf"
        filepath = output_dir / filename
        
        metadata = {
            "model": "Senate AI",
            "file": current_file,
            "total_files": current_file,
            "format": "GGUF F16"
        }
        
        print(f"\n\n💾 Saving {filename} (final)...")
        size_mb, params = write_gguf_file(filepath, all_tensors, metadata)
        
        file_info.append({
            "file": filename,
            "size_mb": round(size_mb, 1),
            "senators": len(all_tensors),
            "params": params
        })
        
        print(f"   ✅ {size_mb:.1f}MB | {len(all_tensors)} senators | {params:,} params")
    
    # Print summary
    total_size = sum(f["size_mb"] for f in file_info)
    total_senators_exported = sum(f["senators"] for f in file_info)
    
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Files: {len(file_info)}")
    print(f"  Total size: {total_size:.0f}MB")
    print(f"  Senators: {total_senators_exported}")
    print(f"  Avg per senator: {total_size/total_senators_exported*1024:.0f}KB")
    print(f"\n  Files:")
    for info in file_info:
        print(f"    {info['file']}: {info['size_mb']:.1f}MB ({info['senators']} senators)")
    
    # Save manifest
    manifest = {
        "model": "Senate AI",
        "total_senators": total_senators_exported,
        "total_size_mb": round(total_size, 1),
        "files": file_info
    }
    
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Manifest saved to gguf_exports/manifest.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-size', type=int, default=300, help='Max file size in MB')
    parser.add_argument('--name', default='senate_ai', help='Output file prefix')
    
    args = parser.parse_args()
    
    export_all_bundles(output_name=args.name, max_size_mb=args.max_size)
