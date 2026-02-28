import os
import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from my_tokenizer import PianoRollTokenizer


def process_single_file_length(file_name, data_dir, patch_h, patch_w):
    """处理单个文件，返回实际token长度"""
    try:
        file_path = os.path.join(data_dir, file_name)
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()

        idx_value = metadata['time_signature_idx']
        if idx_value == 9:
            idx_value = 5  # 2/2拍

        num_measures = metadata['num_measures']

        # 创建tokenizer实例
        tokenizer = PianoRollTokenizer(
            patch_h=patch_h,
            patch_w=patch_w,
            marker_offset=81,
            measures_length=88,
            end_marker_part0=170,
            end_marker_part1=171,
            empty_marker=169,
            img_h=88
        )

        # 实际计算token长度
        total_tokens = 4  # BOS + 拍号 + BPM + EOS

        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']

            # 实际token化过程
            tokens_0 = tokenizer.image_to_patch_tokens(measure[:2], strict_mode=True)
            tokens_1 = tokenizer.image_to_patch_tokens(measure[2:], strict_mode=True)
            compressed_tokens_0 = tokenizer.compress_tokens(tokens_0, end_marker=170)
            compressed_tokens_1 = tokenizer.compress_tokens(tokens_1, end_marker=171)
            measure_tokens = 1 + len(compressed_tokens_0) + len(compressed_tokens_1)  # BAR token + compressed tokens
            total_tokens += measure_tokens

        return file_name, total_tokens, True

    except Exception as e:
        return file_name, 1000, False  # 默认长度1000

def precompute_dataset_lengths(data_dir, patch_h=1, patch_w=4, max_workers=24):
    """
    预计算整个数据集的长度并保存缓存
    
    Args:
        data_dir: 数据目录
        patch_h, patch_w: patch参数
        max_workers: 并行进程数
    """
    cache_file = os.path.join(data_dir, '.lengths_cache.pkl')
    
    # 检查缓存是否已存在
    if os.path.exists(cache_file):
        print(f"缓存文件已存在: {cache_file}")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("跳过预计算")
            return
    
    # 获取所有文件
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    print(f"找到 {len(data_files)} 个文件")
    
    # 准备处理函数
    process_func = partial(process_single_file_length,
                          data_dir=data_dir,
                          patch_h=patch_h,
                          patch_w=patch_w)
    
    # 使用文件名到索引的映射
    file_to_idx = {f: i for i, f in enumerate(data_files)}
    file_lengths = [0] * len(data_files)
    
    print(f"使用 {max_workers} 个进程计算长度...")
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, f): f for f in data_files}
        
        with tqdm(total=len(data_files), desc="计算长度") as pbar:
            for future in as_completed(futures):
                file_name, length, success = future.result()
                idx = file_to_idx[file_name]
                file_lengths[idx] = length
                
                if not success:
                    pbar.set_description(f"⚠ {file_name[:30]}")
                else:
                    pbar.set_description(f"✓ {file_name[:30]} (len={length})")
                
                pbar.update(1)
    
    # 创建排序索引
    sorted_indices = sorted(range(len(data_files)), key=lambda i: file_lengths[i])
    
    # 保存缓存
    cache_data = {
        'data_files': data_files,
        'lengths': file_lengths,
        'sorted_indices': sorted_indices,
        'patch_h': patch_h,
        'patch_w': patch_w
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # 打印统计信息
    lengths_array = np.array(file_lengths)
    print(f"\n长度统计:")
    print(f"  平均: {np.mean(lengths_array):.1f}")
    print(f"  中位数: {np.median(lengths_array):.1f}")
    print(f"  范围: {np.min(lengths_array)} - {np.max(lengths_array)}")
    
    print(f"\n缓存已保存到: {cache_file}")
    print(f"文件大小: {os.path.getsize(cache_file) / 1024 / 1024:.2f} MB")
    
    return cache_data

# 验证缓存
def verify_cache(data_dir):
    """验证缓存文件的完整性"""
    cache_file = os.path.join(data_dir, '.lengths_cache.pkl')
    
    if not os.path.exists(cache_file):
        print("缓存文件不存在")
        return False
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    print(f"缓存信息:")
    print(f"  文件数: {len(cache_data['data_files'])}")
    print(f"  Patch大小: {cache_data['patch_h']}x{cache_data['patch_w']}")
    
    lengths = np.array(cache_data['lengths'])
    print(f"  长度分布:")
    for threshold in [500, 1000, 1500, 2000, 2500, 3000,4000,5000,6000,7000,8000,9000,10000]:
        count = np.sum(lengths <= threshold)
        print(f"    <= {threshold}: {count} ({count/len(lengths)*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    data_dir = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track_optimized"
    
    # 执行预计算
    precompute_dataset_lengths(
        data_dir=data_dir,
        patch_h=1,
        patch_w=4,
        max_workers=32
    )
    
    # 验证缓存
    print("\n验证缓存...")
    verify_cache(data_dir)