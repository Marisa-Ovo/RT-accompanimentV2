import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

def filter_by_length(data_dir, min_length=1000, max_length=6000):
    """
    根据预计算的长度筛选数据集，删除不符合条件的文件
    
    Args:
        data_dir: 数据目录
        min_length: 最小长度
        max_length: 最大长度
    """
    cache_file = os.path.join(data_dir, '.lengths_cache.pkl')
    
    # 检查缓存是否存在
    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        print("请先运行 precompute_dataset_lengths() 生成缓存")
        return
    
    # 读取缓存
    print("读取缓存文件...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    data_files = cache_data['data_files']
    lengths = np.array(cache_data['lengths'])
    
    print(f"\n原始数据集:")
    print(f"  文件数: {len(data_files)}")
    print(f"  长度范围: [{lengths.min()}, {lengths.max()}]")
    print(f"\n筛选条件: 长度 ∈ [{min_length}, {max_length}]")
    
    # 确认操作
    response = input(f"\n⚠️  这将删除不符合条件的文件，是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 筛选
    kept_files = []
    kept_lengths = []
    deleted_files = []
    
    print("\n开始筛选...")
    for i, (filename, length) in enumerate(tqdm(zip(data_files, lengths), total=len(data_files))):
        file_path = os.path.join(data_dir, filename)
        
        if min_length <= length <= max_length:
            # 保留
            kept_files.append(filename)
            kept_lengths.append(length)
        else:
            # 删除
            if os.path.exists(file_path):
                os.remove(file_path)
            deleted_files.append((filename, length))
    
    # 更新缓存
    print("\n更新缓存文件...")
    sorted_indices = sorted(range(len(kept_files)), key=lambda i: kept_lengths[i])
    
    new_cache_data = {
        'data_files': kept_files,
        'lengths': kept_lengths,
        'sorted_indices': sorted_indices,
        'patch_h': cache_data['patch_h'],
        'patch_w': cache_data['patch_w']
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(new_cache_data, f)
    
    # 统计信息
    kept_lengths_array = np.array(kept_lengths)
    
    print(f"\n{'='*60}")
    print(f"筛选完成:")
    print(f"  ✅ 保留: {len(kept_files)} ({len(kept_files)/len(data_files)*100:.1f}%)")
    print(f"  ❌ 删除: {len(deleted_files)} ({len(deleted_files)/len(data_files)*100:.1f}%)")
    
    print(f"\n保留文件的长度统计:")
    print(f"  范围: [{kept_lengths_array.min()}, {kept_lengths_array.max()}]")
    print(f"  平均: {kept_lengths_array.mean():.1f}")
    print(f"  中位数: {np.median(kept_lengths_array):.1f}")
    
    print(f"\n删除原因统计:")
    too_short = sum(1 for _, l in deleted_files if l < min_length)
    too_long = sum(1 for _, l in deleted_files if l > max_length)
    print(f"  长度 < {min_length}: {too_short}")
    print(f"  长度 > {max_length}: {too_long}")
    
    # 显示部分删除的文件
    if deleted_files and len(deleted_files) <= 10:
        print(f"\n删除的文件:")
        for filename, length in deleted_files:
            print(f"  - {filename} (length={length})")
    elif deleted_files:
        print(f"\n删除的文件 (前5个):")
        for filename, length in deleted_files[:5]:
            print(f"  - {filename} (length={length})")
        print(f"  ... 还有 {len(deleted_files)-5} 个")


if __name__ == "__main__":
    data_dir = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_fintune_44"
    
    filter_by_length(
        data_dir=data_dir,
        min_length=1000,
        max_length=6000
    )