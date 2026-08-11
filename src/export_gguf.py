"""
Senate AI - GGUF Exporter
Converts a trained senator bundle to GGUF format for size comparison.
"""

import torch
import sys
import struct
import numpy as np
from pathlib import Path
from collections import OrderedDict


def tensor_to_gguf_type(tensor):
    """Map torch dtype to GGUF type"""
    dtype_map = {
        torch.float32: 0,   # F32
        torch.float16: 1,   # F16
        torch.int32: 4,     # I32
        torch.int16: 5,     # I16
        torch.int8: 7,      # I8
    }
    return dtype_map.get(tensor.dtype, 0)


def write_gguf_header(f, num_tensors, metadata=None):
    """Write GGUF magic and header"""
    # Magic
    f.write(b'GGUF')
    
    # Version 3
    f.write(struct.pack('<I', 3))
    
    # Number of tensors
    f.write(struct.pack('<Q', num_tensors))
    
    # Metadata key count (minimal)
    f.write(struct.pack('<Q', 0))
    
    return f.tell()


def write_tensor_info(f, name, tensor, offset):
    """Write tensor info entry"""
    # Name
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<Q', len(name_bytes)))
    f.write(name_bytes)
    
    # Dimensions
    shape = tensor.shape
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<Q', dim))
    
    # Type
    f.write(struct.pack('<I', tensor_to_gguf_type(tensor)))
    
    # Offset (will be patched)
    f.write(struct.pack('<Q', offset))


def convert_to_f16(tensor):
    """Convert tensor to float16"""
    return tensor.half()


def convert_to_q4_0(tensor):
    """Simple 4-bit quantization (GGUF Q4_0 style)"""
    if tensor.dim() < 2 or tensor.numel() < 32:
        return tensor.half()
    
    # Reshape to 2D for block quantization
    original_shape = tensor.shape
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    
    rows, cols = tensor.shape
    block_size = 32
    
    # Pad columns
    padded_cols = ((cols + block_size - 1) // block_size) * block_size
    padded = torch.zeros(rows, padded_cols, dtype=tensor.dtype)
    padded[:, :cols] = tensor
    
    quantized = torch.zeros(rows, padded_cols // 2 + rows * 2, dtype=torch.float16)
    
    for r in range(rows):
        row = padded[r]
        
        # Find scale (max abs value)
        max_val = row.abs().max()
        if max_val == 0:
            max_val = 1.0
        scale = max_val / 7.0
        
        # Store scale as float16
        quantized[r, 0] = scale
        
        # Quantize each block
        for b in range(0, padded_cols, block_size):
            block = row[b:b+block_size]
            
            # Quantize to 4-bit
            q_block = torch.clamp(torch.round(block / scale), -8, 7).to(torch.int8)
            
            # Pack 4-bit values
            byte_offset = 1 + (b // 2)
            for i in range(0, len(q_block), 2):
                if i + 1 < len(q_block):
                    packed = ((q_block[i].item() + 8) & 0xF) | (((q_block[i+1].item() + 8) & 0xF) << 4)
                else:
                    packed = ((q_block[i].item() + 8) & 0xF)
                
                if byte_offset + i//2 < quantized.shape[1]:
                    quantized[r, byte_offset + i//2] = packed
    
    return quantized


def export_senator_gguf(senator_id, quantization='f16'):
    """Export a single senator to GGUF"""
    
    # Load senator from bundle
    bundle_id = senator_id // 45
    bundle_path = f"senate_bundles/bundle_{bundle_id:03d}.pt"
    
    if not Path(bundle_path).exists():
        print(f"❌ Bundle {bundle_id} not found")
        return
    
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    senators = bundle.get('senators', {})
    
    senator_key = str(senator_id)
    if senator_key not in senators:
        print(f"❌ Senator {senator_id} not found in bundle {bundle_id}")
        return
    
    data = senators[senator_key]
    state_dict = data.get('state_dict', data)
    config = data.get('config', {})
    
    # Create output
    output_dir = Path("gguf_exports")
    output_dir.mkdir(exist_ok=True)
    
    qname = {'f16': 'f16', 'q4': 'Q4_0'}.get(quantization, 'f16')
    output_path = output_dir / f"senator_{senator_id:04d}_{qname}.gguf"
    
    print(f"\nExporting Senator {senator_id} ({config.get('specialties', [])})")
    print(f"Quantization: {quantization}")
    
    # Process tensors
    tensors = OrderedDict()
    total_params = 0
    
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        total_params += tensor.numel()
        
        if quantization == 'q4' and tensor.dim() >= 2 and tensor.numel() >= 32:
            tensors[name] = convert_to_q4_0(tensor)
        elif quantization == 'f16':
            tensors[name] = convert_to_f16(tensor)
        else:
            tensors[name] = tensor
    
    print(f"Parameters: {total_params:,}")
    
    # Write GGUF file
    with open(output_path, 'wb') as f:
        # Header
        write_gguf_header(f, len(tensors))
        
        # Calculate offsets
        header_end = f.tell()
        tensor_info_start = header_end
        
        # Tensor info size: each entry is ~32 bytes + name
        tensor_info_size = len(tensors) * 64
        tensor_data_start = tensor_info_start + tensor_info_size
        
        # Write tensor info placeholders
        current_offset = tensor_data_start
        tensor_offsets = []
        
        for name, tensor in tensors.items():
            write_tensor_info(f, name, tensor, current_offset)
            tensor_offsets.append(current_offset)
            
            # Calculate tensor data size
            if tensor.dtype == torch.float16:
                current_offset += tensor.numel() * 2
            elif tensor.dtype == torch.int8:
                current_offset += tensor.numel()
            else:
                current_offset += tensor.numel() * 4
            
            # Align to 32 bytes
            current_offset = ((current_offset + 31) // 32) * 32
        
        # Write tensor data
        for (name, tensor), offset in zip(tensors.items(), tensor_offsets):
            # Pad to offset
            f.seek(offset)
            
            # Write raw bytes
            if tensor.dtype == torch.float16:
                f.write(tensor.numpy().tobytes())
            elif tensor.dtype == torch.int8:
                f.write(tensor.numpy().astype(np.int8).tobytes())
            else:
                f.write(tensor.numpy().tobytes())
    
    # Get size
    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_kb = size_bytes / 1024
    
    print(f"GGUF size: {size_mb:.2f}MB ({size_kb:.0f}KB)")
    
    # Compare to original
    original_params_mb = total_params * 4 / (1024 * 1024)
    compression = (1 - size_mb / original_params_mb) * 100 if original_params_mb > 0 else 0
    print(f"Original (f32): {original_params_mb:.1f}MB")
    print(f"Compression: {compression:.1f}%")
    print(f"Saved to: {output_path}")
    
    return size_mb


def export_bundle_gguf(bundle_id, quantization='f16'):
    """Export all senators in a bundle"""
    start_id = bundle_id * 45
    end_id = start_id + 45
    
    total_size = 0
    print(f"\n{'='*50}")
    print(f"  EXPORTING BUNDLE {bundle_id} ({quantization})")
    print(f"  Senators {start_id}-{end_id-1}")
    print(f"{'='*50}")
    
    for senator_id in range(start_id, end_id):
        size = export_senator_gguf(senator_id, quantization)
        if size:
            total_size += size
    
    print(f"\n{'='*50}")
    print(f"  Bundle {bundle_id} total: {total_size:.1f}MB")
    print(f"  Per senator avg: {total_size/45:.2f}MB")
    print(f"{'='*50}")
    
    return total_size


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('target', help='Senator ID or Bundle ID (prefixed with b)')
    parser.add_argument('--quant', default='f16', choices=['f16', 'q4'], help='Quantization')
    
    args = parser.parse_args()
    
    if args.target.startswith('b'):
        bundle_id = int(args.target[1:])
        export_bundle_gguf(bundle_id, args.quant)
    else:
        senator_id = int(args.target)
        export_senator_gguf(senator_id, args.quant)
