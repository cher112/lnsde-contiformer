"""
Data loading and preprocessing utilities for light curves
支持lombscaler折叠光变曲线数据的加载和预处理
"""

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LightCurveDataset(Dataset):
    """
    光变曲线数据集类
    支持ASAS, LINEAR, MACHO三个数据集的lombscaler折叠数据
    """
    def __init__(self, data_path: str, normalize: bool = False,  # 默认关闭归一化
                 max_seq_len: int = 512, feature_keys: List[str] = None):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径(.pkl)
            normalize: 是否标准化特征
            max_seq_len: 最大序列长度
            feature_keys: 使用的特征键，默认['time', 'mag', 'errmag']
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        
        if feature_keys is None:
            self.feature_keys = ['time', 'mag', 'errmag']
        else:
            self.feature_keys = feature_keys
            
        # 加载数据
        print(f"正在加载并预处理数据: {data_path}")
        raw_data = self._load_data()
        self.data = self._preprocess_data(raw_data)  # 预处理数据
        self.num_samples = len(self.data)
        
        # 提取标签和类别信息
        self.labels = [item['label'] for item in self.data]
        self.class_names = sorted(list(set([item['class_name'] for item in self.data])))
        self.num_classes = len(self.class_names)
        
        # 标签编码器
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # 计算类别权重（用于处理不平衡数据）
        self.class_weights = self._compute_class_weights()
        
        # 特征标准化器
        if self.normalize:
            self.scalers = self._fit_scalers()
        else:
            self.scalers = None
            
        print(f"数据集预处理完成: {self.num_samples}个样本, {self.num_classes}个类别")
        self._print_dataset_info()
    
    def _load_data(self) -> List[Dict]:
        """加载lombscaler折叠数据"""
        print(f"正在加载数据: {self.data_path}")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
        if not isinstance(data, list):
            raise ValueError(f"数据格式错误: 期望list, 得到{type(data)}")
            
        print(f"原始数据包含 {len(data)} 个样本")
        return data
    
    def _preprocess_data(self, raw_data: List[Dict]) -> List[Dict]:
        """预处理所有数据，避免在__getitem__中重复计算"""
        processed_data = []
        print("正在预处理时间序列数据...")
        
        for i, item in enumerate(raw_data):
            try:
                # 提取基础信息
                mask = item['mask'].astype(bool)
                times = item['time'].astype(np.float32)
                mags = item['mag'].astype(np.float32)
                errmags = item['errmag'].astype(np.float32)
                period = np.float32(item['period'])
                
                # 获取有效的时间点和数据
                valid_indices = mask
                valid_times = times[valid_indices]
                valid_mags = mags[valid_indices]
                valid_errmags = errmags[valid_indices]
                
                # 按时间排序
                sort_indices = np.argsort(valid_times)
                sorted_times = valid_times[sort_indices]
                sorted_mags = valid_mags[sort_indices]
                sorted_errmags = valid_errmags[sort_indices]
                
                # 确保时间严格递增（去除重复时间点）
                unique_indices = np.diff(sorted_times, prepend=-np.inf) > 1e-6
                final_times = sorted_times[unique_indices]
                final_mags = sorted_mags[unique_indices]
                final_errmags = sorted_errmags[unique_indices]
                
                actual_length = len(final_times)
                
                # Padding到固定长度
                max_len = len(times)
                padded_times = np.zeros(max_len, dtype=np.float32)
                padded_mags = np.zeros(max_len, dtype=np.float32)
                padded_errmags = np.zeros(max_len, dtype=np.float32)
                padded_mask = np.zeros(max_len, dtype=bool)
                
                padded_times[:actual_length] = final_times
                padded_mags[:actual_length] = final_mags
                padded_errmags[:actual_length] = final_errmags
                padded_mask[:actual_length] = True
                
                # 构建预处理后的数据项
                processed_item = {
                    'times': padded_times,
                    'mags': padded_mags,
                    'errmags': padded_errmags,
                    'mask': padded_mask,
                    'period': period,
                    'actual_length': actual_length,
                    'label': item['label'],
                    'class_name': item['class_name'],
                    'original_index': i
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"预处理第{i}个样本时出错: {e}")
                continue
                
        print(f"预处理完成: {len(processed_data)}/{len(raw_data)} 个样本")
        return processed_data
    
    def _compute_class_weights(self) -> torch.Tensor:
        """计算类别权重，用于处理不平衡数据"""
        from collections import Counter
        
        label_counts = Counter(self.encoded_labels)
        total_samples = len(self.encoded_labels)
        num_classes = len(label_counts)
        
        weights = torch.zeros(num_classes)
        for label, count in label_counts.items():
            weights[label] = total_samples / (num_classes * count)
            
        return weights
    
    def _fit_scalers(self) -> Dict[str, StandardScaler]:
        """拟合特征标准化器"""
        scalers = {}
        
        # 收集所有特征用于拟合标准化器
        all_features = {key: [] for key in self.feature_keys}
        
        for item in self.data:
            mask = item['mask'].astype(bool)
            for key in self.feature_keys:
                if key in item:
                    feature_data = item[key][mask]  # 只使用有效数据点
                    all_features[key].extend(feature_data.tolist())
        
        # 拟合标准化器
        for key in self.feature_keys:
            if all_features[key]:  # 确保有数据
                scaler = StandardScaler()
                scaler.fit(np.array(all_features[key]).reshape(-1, 1))
                scalers[key] = scaler
            else:
                print(f"警告: 特征 {key} 没有有效数据")
                scalers[key] = None
                
        return scalers
    
    def _normalize_features(self, item: Dict) -> np.ndarray:
        """标准化特征"""
        features = []
        
        for key in self.feature_keys:
            if key in item and self.scalers.get(key) is not None:
                feature_data = item[key].reshape(-1, 1)
                normalized_data = self.scalers[key].transform(feature_data).flatten()
                features.append(normalized_data)
            elif key in item:
                # 如果没有标准化器，直接使用原始数据
                features.append(item[key])
            else:
                # 如果特征不存在，用零填充
                features.append(np.zeros(len(item['time'])))
                
        return np.column_stack(features)  # (seq_len, num_features)
    
    def _print_dataset_info(self):
        """打印数据集信息（简化版）"""
        print(f"=== 数据集 ===")
        print(f"样本: {self.num_samples}, 类别: {self.num_classes}, 特征: {len(self.feature_keys)}, 序列长度: {self.max_seq_len}")
        
        # 简化类别分布
        from collections import Counter
        label_counts = Counter(self.labels)
        class_info = []
        for i, class_name in enumerate(self.class_names):
            original_label = self.label_encoder.classes_[i]
            count = sum(1 for label in self.labels if label == original_label)
            class_info.append(f"{class_name}: {count}")
        print(f"类别分布: {', '.join(class_info)}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本 - 使用预处理数据，大幅提升性能
        Returns:
            sample: {
                'x': (seq_len, 2) 原始光变曲线 [time, mag]
                'time_steps': (seq_len,) 时间序列  
                'periods': (1,) 周期信息
                'attn_mask': (seq_len,) 注意力mask
                'label': 类别标签
                'class_name': 类别名称
                'valid_length': 有效序列长度
            }
        """
        item = self.data[idx]
        
        # 直接从预处理数据中获取，无需重复计算
        times = item['times']
        mags = item['mags'] 
        errmags = item['errmags']
        mask = item['mask']
        period = item['period']
        actual_length = item['actual_length']
        
        # 构建 elses 格式的输入数据
        x = np.column_stack([times, mags])
        time_steps = times
        periods = np.array([period], dtype=np.float32)
        attn_mask = mask.astype(np.float32)
        
        # 转换为tensor
        x = torch.from_numpy(x.astype(np.float32))
        time_steps = torch.from_numpy(time_steps)
        periods = torch.from_numpy(periods)
        attn_mask = torch.from_numpy(attn_mask)
        
        # 获取标签
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        return {
            'x': x,                    # elses 主要输入 [seq_len, 2]
            'time_steps': time_steps,  # elses 时间步长 [seq_len]
            'periods': periods,        # elses 周期 [1]
            'attn_mask': attn_mask,   # elses 注意力mask [seq_len]
            'label': label,
            'class_name': item['class_name'],
            'valid_length': actual_length,
            'original_index': item['original_index'],
            # 保留原有格式兼容
            'features': torch.cat([time_steps.unsqueeze(1), 
                                  torch.from_numpy(mags.astype(np.float32)).unsqueeze(1),
                                  torch.from_numpy(errmags.astype(np.float32)).unsqueeze(1)], dim=1),
            'times': time_steps,
            'mask': torch.from_numpy(mask)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批处理整理函数，处理不等长序列，支持elses格式
    """
    batch_size = len(batch)
    max_len = 512  # 固定长度为512
    
    # 检查是否使用新的elses格式
    if 'x' in batch[0]:
        # 新的elses兼容格式
        x_dim = batch[0]['x'].size(1)  # 应该是2 [time, mag]
        
        # 初始化elses格式的批处理张量
        x = torch.zeros(batch_size, max_len, x_dim)
        time_steps = torch.zeros(batch_size, max_len)
        periods = torch.zeros(batch_size, 1)
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
        labels = torch.zeros(batch_size, dtype=torch.long)
        valid_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # 同时保持原有格式兼容
        features = torch.zeros(batch_size, max_len, 3)  # [time, mag, errmag]
        times = torch.zeros(batch_size, max_len)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        # 填充数据
        for i, item in enumerate(batch):
            seq_len = min(item['x'].size(0), max_len)  # 确保不超过512
            
            # elses格式
            x[i, :seq_len] = item['x'][:seq_len]
            time_steps[i, :seq_len] = item['time_steps'][:seq_len]
            periods[i] = item['periods']
            attn_mask[i, :seq_len] = item['attn_mask'][:seq_len]
            labels[i] = item['label']
            valid_lengths[i] = seq_len
            
            # 兼容格式
            features[i, :seq_len] = item['features'][:seq_len]
            times[i, :seq_len] = item['times'][:seq_len] 
            masks[i, :seq_len] = item['mask'][:seq_len]
        
        return {
            # elses格式输出
            'x': x,                      # (batch, 512, 2)
            'time_steps': time_steps,    # (batch, 512)  
            'periods': periods,          # (batch, 1)
            'attn_mask': attn_mask,     # (batch, 512) - float类型的mask
            'labels': labels,           # (batch,)
            'valid_lengths': valid_lengths,
            'class_names': [item['class_name'] for item in batch],
            # 兼容格式输出  
            'features': features,       # (batch, 512, 3)
            'times': times,            # (batch, 512)
            'mask': masks              # (batch, 512) - bool类型的mask
        }
    
    else:
        # 原有格式处理
        feature_dim = batch[0]['features'].size(1)
        
        # 初始化批处理张量
        features = torch.zeros(batch_size, max_len, feature_dim)
        times = torch.zeros(batch_size, max_len)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        labels = torch.zeros(batch_size, dtype=torch.long)
        valid_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # 填充数据
        for i, item in enumerate(batch):
            seq_len = min(item['features'].size(0), max_len)
            
            features[i, :seq_len] = item['features'][:seq_len]
            times[i, :seq_len] = item['times'][:seq_len]
            masks[i, :seq_len] = item['mask'][:seq_len]
            labels[i] = item['label']
            valid_lengths[i] = seq_len
        
        return {
            'features': features,
            'times': times,
            'mask': masks,
            'labels': labels,
            'valid_lengths': valid_lengths,
            'class_names': [item['class_name'] for item in batch]
        }


def create_dataloaders(data_path: str, 
                      batch_size: int = 32,
                      train_ratio: float = 0.8,
                      test_ratio: float = 0.2,
                      normalize: bool = False,  # 默认关闭归一化，因为数据已经folded
                      num_workers: int = 8,
                      random_seed: int = 42) -> Tuple[DataLoader, DataLoader, int]:
    """
    创建训练、测试数据加载器（80-20分割）
    Args:
        data_path: 数据文件路径
        batch_size: 批大小
        train_ratio: 训练集比例  
        test_ratio: 测试集比例
        normalize: 是否标准化
        num_workers: 数据加载进程数
        random_seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 创建完整数据集
    full_dataset = LightCurveDataset(
        data_path=data_path,
        normalize=normalize
    )
    
    # 计算分割大小
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    print(f"数据集分割:")
    print(f"  训练集: {train_size} 样本 ({train_ratio*100:.1f}%)")
    print(f"  测试集: {test_size} 样本 ({test_ratio*100:.1f}%)")
    
    # 随机分割数据集
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 创建数据加载器 - 优化GPU利用率
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程，减少重启开销
        prefetch_factor=8,        # 增加预取批次数
        drop_last=True           # 丢弃不完整的batch，提高训练稳定性
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )
    
    return train_loader, test_loader, full_dataset.num_classes