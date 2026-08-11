"""
Senate AI - 4-Stage Model Optimizer
1. Smart Prune (redistribute weak to strong)
2. Safe Precision Prune (merge insignificant, same row only)  
3. Progressive Bit Reduction (changed weights only, with timeout)
4. 8-bit Quantize
"""

import torch
import torch.nn as nn
import time
import os
import sys
from pathlib import Path


def progress_bar(done, total, label="", width=30):
    pct = done / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f"   {label} [{bar}] {pct*100:.0f}%"


class SenateOptimizer:
    """Optimizes a Senate bundle through 4 stages"""
    
    def __init__(self, bundle_path, output_path=None):
        self.bundle_path = bundle_path
        self.output_path = output_path or bundle_path
        
        print(f"\n{'='*50}")
        print(f"🔧 SENATE OPTIMIZER")
        print(f"{'='*50}")
        print(f"   Bundle: {bundle_path}")
        
        self.bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
        self.senators = self.bundle.get('senators', {})
        print(f"   Senators: {len(self.senators)}")
    
    def get_bundle_size(self):
        total_params = 0
        non_zero = 0
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            for param in state_dict.values():
                if isinstance(param, torch.Tensor):
                    total_params += param.numel()
                    non_zero += (param != 0).sum().item()
        
        sparsity = (1 - non_zero / max(total_params, 1)) * 100
        size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'non_zero': non_zero,
            'sparsity': sparsity,
            'size_mb': size_mb
        }
    
    def smart_prune(self, target_sparsity=0.5):
        print(f"\n🧠 STAGE 1: SMART PRUNE (target: {target_sparsity*100:.0f}%)")
        
        total_redistributed = 0
        total_senators = len(self.senators)
        
        for idx, (senator_id, data) in enumerate(self.senators.items()):
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                
                for row_idx in range(weight.shape[0]):
                    row = weight[row_idx]
                    abs_row = row.abs()
                    
                    if abs_row.sum() == 0:
                        continue
                    
                    k = max(1, int((1 - target_sparsity) * len(row)))
                    if k >= len(row):
                        continue
                    
                    threshold = torch.kthvalue(abs_row, len(row) - k).values
                    strong_mask = abs_row >= threshold
                    strong_idx = torch.where(strong_mask)[0]
                    weak_idx = torch.where(~strong_mask)[0]
                    
                    if len(strong_idx) == 0 or len(weak_idx) == 0:
                        continue
                    
                    for wi in weak_idx:
                        weak_val = row[wi]
                        if abs(weak_val) < 0.00001:
                            row[wi] = 0
                            continue
                        
                        best_si = strong_idx[0]
                        best_sim = -999
                        
                        for si in strong_idx[:min(20, len(strong_idx))]:
                            sign_match = 1 if (weak_val * row[si]) > 0 else -1
                            sim = sign_match * (1 - min(abs(weak_val - row[si].abs()), 1.0))
                            if sim > best_sim:
                                best_sim = sim
                                best_si = si
                        
                        row[best_si] += weak_val * 0.6
                        row[wi] = 0
                        total_redistributed += 1
                
                param.data = weight.to(param.dtype)
            
            if (idx + 1) % 10 == 0:
                print(f"\r{progress_bar(idx+1, total_senators, 'Smart Prune')}", end='')
        
        print(f"\r{progress_bar(total_senators, total_senators, 'Smart Prune')}")
        print(f"   Redistributed: {total_redistributed:,} weights")
        return total_redistributed
    
    def precision_prune_safe(self, significance=2):
        print(f"\n🎯 STAGE 2: SAFE PRECISION PRUNE (sig={significance})")
        
        total_merged = 0
        total_senators = len(self.senators)
        
        for idx, (senator_id, data) in enumerate(self.senators.items()):
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                
                for row_idx in range(weight.shape[0]):
                    row = weight[row_idx]
                    abs_row = row.abs()
                    
                    if abs_row.sum() == 0:
                        continue
                    
                    threshold = torch.kthvalue(abs_row, int(0.5 * len(row))).values
                    strong_mask = abs_row >= threshold
                    strong_idx = torch.where(strong_mask)[0]
                    
                    if len(strong_idx) < 2:
                        continue
                    
                    for col_idx in range(len(row)):
                        if strong_mask[col_idx]:
                            continue
                        
                        val = row[col_idx]
                        if abs(val) < 0.0001:
                            row[col_idx] = 0
                            continue
                        
                        rounded = round(val.item(), significance)
                        if abs(val - rounded) < (10 ** -(significance + 1)):
                            best_si = strong_idx[0]
                            best_dist = float('inf')
                            
                            for si in strong_idx:
                                dist = abs(val - row[si].item())
                                if dist < best_dist:
                                    best_dist = dist
                                    best_si = si
                            
                            row[best_si] += val * 0.7
                            row[col_idx] = 0
                            total_merged += 1
                
                param.data = weight.to(param.dtype)
            
            if (idx + 1) % 10 == 0:
                print(f"\r{progress_bar(idx+1, total_senators, 'Precision')}", end='')
        
        print(f"\r{progress_bar(total_senators, total_senators, 'Precision')}")
        print(f"   Merged: {total_merged:,}")
        return total_merged
    
    def progressive_bit_reduce(self, timeout_minutes=25, margin=0.1):
        print(f"\n📉 STAGE 3: PROGRESSIVE BITS (timeout: {timeout_minutes}min)")
        
        nodes_8bit = nodes_7bit = nodes_6bit = nodes_merged = 0
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        stopped_early = False
        total_senators = len(self.senators)
        
        for idx, (senator_id, data) in enumerate(self.senators.items()):
            if time.time() - start_time > timeout_seconds:
                stopped_early = True
                break
            
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                
                for row_idx in range(weight.shape[0]):
                    if time.time() - start_time > timeout_seconds:
                        stopped_early = True
                        break
                    
                    row = weight[row_idx]
                    row_max = row.abs().max()
                    
                    if row_max == 0:
                        continue
                    
                    upper_bound = row_max * (0.3 + margin)
                    lower_bound = row_max * 0.001
                    
                    for col_idx in range(len(row)):
                        val = row[col_idx]
                        abs_val = abs(val)
                        
                        if abs_val < 0.0001:
                            row[col_idx] = 0
                            nodes_merged += 1
                            continue
                        
                        if abs_val > upper_bound or abs_val < lower_bound:
                            nodes_8bit += 1
                            continue
                        
                        val_7bit = round(val.item() * 127) / 127
                        val_6bit = round(val.item() * 63) / 63
                        
                        if abs(val - val_6bit) < 0.001:
                            row[col_idx] = val_6bit
                            nodes_6bit += 1
                        elif abs(val - val_7bit) < 0.0001:
                            row[col_idx] = val_7bit
                            nodes_7bit += 1
                        else:
                            nodes_8bit += 1
                    
                    param.data = weight.to(param.dtype)
                
                if stopped_early:
                    break
            
            if (idx + 1) % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"\r{progress_bar(idx+1, total_senators, f'Bits ({elapsed:.0f}min)')}", end='')
        
        elapsed = (time.time() - start_time) / 60
        total = nodes_8bit + nodes_7bit + nodes_6bit + nodes_merged
        eff_bits = (nodes_8bit * 8 + nodes_7bit * 7 + nodes_6bit * 6) / max(total, 1)
        
        print(f"\r{progress_bar(total_senators, total_senators, 'Bits')}")
        print(f"   8-bit: {nodes_8bit:,}  7-bit: {nodes_7bit:,}  6-bit: {nodes_6bit:,}  merged: {nodes_merged:,}")
        print(f"   Effective: {eff_bits:.1f}-bit  |  Time: {elapsed:.1f}min")
        
        if stopped_early:
            print(f"   ⏰ Timeout — saved progress")
        
        return eff_bits
    
    def quantize_senators(self):
        print(f"\n🔧 STAGE 4: 8-BIT QUANTIZE")
        
        quantized_count = 0
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if isinstance(param, torch.Tensor) and param.dtype == torch.float32:
                    if param.numel() > 100 and param.abs().max() > 0:
                        state_dict[param_name] = param.half()
                        quantized_count += 1
        
        print(f"   Quantized: {quantized_count} tensors to float16")
        return quantized_count
    
    def optimize(self, target_sparsity=0.5, timeout_minutes=25):
        before = self.get_bundle_size()
        print(f"\n📊 Before: {before['total_params']:,} params, {before['size_mb']:.1f}MB")
        
        self.smart_prune(target_sparsity=target_sparsity)
        self.precision_prune_safe(significance=2)
        self.progressive_bit_reduce(timeout_minutes=timeout_minutes)
        self.quantize_senators()
        
        after = self.get_bundle_size()
        saved_mb = before['size_mb'] - after['size_mb']
        
        print(f"\n{'='*50}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"  Before: {before['size_mb']:.1f}MB")
        print(f"  After:  {after['size_mb']:.1f}MB")
        print(f"  Saved:  {saved_mb:.1f}MB ({saved_mb/before['size_mb']*100:.0f}%)")
        print(f"  Sparsity: {after['sparsity']:.1f}%")
        
        return after
    
    def save(self):
        print(f"\n💾 Saving optimized bundle...")
        torch.save(self.bundle, self.output_path)
        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"   Saved: {self.output_path} ({size_mb:.1f}MB)")
        return size_mb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle_id', type=int)
    parser.add_argument('target_sparsity', nargs='?', type=float, default=0.5)
    
    args = parser.parse_args()
    
    bundle_path = f"senate_bundles/bundle_{args.bundle_id:03d}.pt"
    
    if not Path(bundle_path).exists():
        print(f"❌ Bundle {args.bundle_id} not found at {bundle_path}")
        sys.exit(1)
    
    optimizer = SenateOptimizer(bundle_path)
    optimizer.optimize(target_sparsity=args.target_sparsity)
    optimizer.save()
