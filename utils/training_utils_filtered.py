"""
改进的训练工具函数 - 实现样本级别的NaN过滤
"""

import torch
import numpy as np
from torch.cuda.amp import autocast
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, confusion_matrix


def filter_nan_samples(features, y, mask, times=None, model=None, model_type='linear_noise', dataset_config=None, epoch=0, batch_idx=0):
    """
    检测并过滤批次中导致NaN的样本
    返回清理后的批次数据
    """
    if model is None or dataset_config is None:
        return features, y, mask, times
    
    batch_size = len(y)
    valid_indices = []
    nan_indices = []
    
    # 逐个检测每个样本
    for i in range(batch_size):
        single_features = features[i:i+1]
        single_mask = mask[i:i+1] 
        single_y = y[i:i+1]
        single_times = times[i:i+1] if times is not None else None
        
        try:
            with torch.no_grad():
                # 根据模型类型进行前向传播
                if model_type in ['geometric', 'langevin']:
                    single_logits, single_sde_features = model(single_features, single_times, single_mask)
                    single_loss = model.compute_loss(
                        single_logits, single_y, single_sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                else:  # linear_noise
                    single_logits = model(single_features, single_mask)
                    single_loss = model.compute_loss(
                        single_logits, single_y, None,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                
                if isinstance(single_loss, tuple):
                    single_loss = single_loss[0]
                
                # 检查是否产生NaN或Inf
                if torch.isnan(single_loss) or torch.isinf(single_loss):
                    nan_indices.append(i)
                    # 静默记录到日志文件
                    with open('nan_samples.log', 'a') as f:
                        f.write(f"Epoch {epoch}, Batch {batch_idx}, Sample {i} filtered out (NaN loss)\n")
                else:
                    valid_indices.append(i)
                    
        except Exception as e:
            # 如果出现任何异常，也认为是有问题的样本
            nan_indices.append(i)
            with open('nan_samples.log', 'a') as f:
                f.write(f"Epoch {epoch}, Batch {batch_idx}, Sample {i} filtered out (Exception: {str(e)[:50]})\n")
    
    # 如果所有样本都有问题，返回空批次
    if len(valid_indices) == 0:
        return None, None, None, None
    
    # 如果有样本被过滤，重新组织批次
    if len(nan_indices) > 0:
        valid_indices = torch.tensor(valid_indices, device=features.device)
        filtered_features = features[valid_indices]
        filtered_y = y[valid_indices]
        filtered_mask = mask[valid_indices]
        filtered_times = times[valid_indices] if times is not None else None
        
        return filtered_features, filtered_y, filtered_mask, filtered_times
    
    return features, y, mask, times


def train_epoch_with_filtering(model, train_loader, optimizer, criterion, device, model_type, dataset_config, scaler=None, gradient_accumulation_steps=1, epoch=0, gradient_clip=0.5):
    """训练一个epoch - 支持样本级别的NaN过滤"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_filtered = 0  # 被过滤的样本总数
    
    all_predictions = []
    all_labels = []
    all_periods = []
    all_probas = []
    
    # 创建进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training",
                ncols=140, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')
    
    accumulated_loss = 0.0
    
    for batch_idx, batch_data in pbar:
        # 数据加载器返回字典格式
        if isinstance(batch_data, dict):
            # 从字典中提取数据
            features = batch_data['features']  # (batch, seq_len, 3) [time, mag, errmag]
            y = batch_data['labels'] if 'labels' in batch_data else batch_data['label']
            mask = batch_data['mask']  # (batch, seq_len) bool mask
            periods = batch_data['periods']  # (batch, 1)
            
            # 检查是否有times数据（用于geometric/langevin模型）
            if 'times' in batch_data:
                times = batch_data['times']  # (batch, seq_len)
            else:
                times = None
        else:
            # 兼容旧的元组格式
            if len(batch_data) == 4:  # (features, y, mask, periods)
                features, y, mask, periods = batch_data
                times = None
            else:  # (features, times, y, mask, periods) 
                features, times, y, mask, periods = batch_data
        
        features = features.to(device)
        y = y.to(device)
        mask = mask.to(device)
        periods = periods.to(device)
        
        original_batch_size = len(y)
        
        # 步骤1: 过滤导致NaN的样本
        filtered_features, filtered_y, filtered_mask, filtered_times = filter_nan_samples(
            features, y, mask, times, model, model_type, dataset_config, epoch, batch_idx
        )
        
        # 如果整个批次都被过滤掉了，跳过
        if filtered_features is None:
            total_filtered += original_batch_size
            continue
        
        filtered_batch_size = len(filtered_y)
        samples_filtered = original_batch_size - filtered_batch_size
        total_filtered += samples_filtered
        
        # 步骤2: 使用过滤后的样本进行训练
        if scaler:  # 混合精度训练
            with autocast():
                # 根据模型类型进行前向传播
                if model_type == 'geometric':
                    logits, sde_features = model(filtered_features, filtered_times, filtered_mask)
                    loss = model.compute_loss(
                        logits, filtered_y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )[0]
                elif model_type == 'langevin':
                    logits, sde_features = model(filtered_features, filtered_times, filtered_mask)
                    loss = model.compute_loss(
                        logits, filtered_y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]
                else:  # linear_noise
                    logits = model(filtered_features, filtered_mask)
                    sde_features = None
                    loss = model.compute_loss(
                        logits, filtered_y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]
            
            # 现在loss应该是正常的，但为了安全还是检查一下
            if torch.isnan(loss) or torch.isinf(loss):
                # 如果过滤后仍然有NaN，跳过这个批次
                continue
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            scaler.scale(loss).backward()
            
            # 梯度更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                
                # 检查梯度
                has_nan_inf = False
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_inf = True
                            break
                
                if not has_nan_inf:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    batch_loss = accumulated_loss * gradient_accumulation_steps
                else:
                    optimizer.zero_grad()
                    scaler.update()
                    batch_loss = 0.0
                
                accumulated_loss = 0.0
            else:
                batch_loss = loss.item() * gradient_accumulation_steps
                
        else:  # 普通训练
            # 类似的逻辑，但不使用autocast
            if model_type == 'geometric':
                logits, sde_features = model(filtered_features, filtered_times, filtered_mask)
                loss = model.compute_loss(
                    logits, filtered_y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )[0]
            elif model_type == 'langevin':
                logits, sde_features = model(filtered_features, filtered_times, filtered_mask)
                loss = model.compute_loss(
                    logits, filtered_y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]
            else:  # linear_noise
                logits = model(filtered_features, filtered_mask)
                sde_features = None
                loss = model.compute_loss(
                    logits, filtered_y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                has_nan_inf = False
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_inf = True
                            break
                
                if not has_nan_inf:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = accumulated_loss * gradient_accumulation_steps
                else:
                    optimizer.zero_grad()
                    batch_loss = 0.0
                
                accumulated_loss = 0.0
            else:
                batch_loss = loss.item() * gradient_accumulation_steps
        
        # 统计信息
        total_loss += batch_loss
        with torch.no_grad():
            probas = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            total += filtered_y.size(0)
            correct += (predicted == filtered_y).sum().item()
            
            # 收集用于计算F1等指标的数据
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(filtered_y.cpu().numpy())
            all_periods.extend(periods[:filtered_batch_size].cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
        
        # 更新进度条
        current_acc = 100.0 * correct / total if total > 0 else 0
        avg_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else batch_loss
        postfix = f'Loss: {avg_loss:.4f}, Acc: {current_acc:.2f}%'
        if samples_filtered > 0:
            postfix += f', Filtered: {samples_filtered}'
        pbar.set_postfix_str(postfix)
    
    pbar.close()
    
    # 计算最终指标
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    # 计算各类别准确率和F1分数
    if len(all_predictions) > 0:
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # 各类别准确率
        unique_labels = sorted(set(all_labels))
        class_acc = {}
        for label in unique_labels:
            mask_label = np.array(all_labels) == label
            if mask_label.sum() > 0:
                class_acc[label] = np.mean(np.array(all_predictions)[mask_label] == label) * 100
        
        metrics = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted, 
            'recall': recall,
            'total_samples': total,
            'filtered_samples': total_filtered
        }
    else:
        class_acc = {}
        metrics = {'total_samples': 0, 'filtered_samples': total_filtered}
    
    if total_filtered > 0:
        print(f"\n本轮训练过滤掉 {total_filtered} 个导致NaN的样本")
    
    return avg_loss, accuracy, class_acc, metrics