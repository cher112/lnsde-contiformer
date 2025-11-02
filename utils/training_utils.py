"""
训练相关工具函数 - 简化版，专注于稳定性和NaN处理
"""

import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, confusion_matrix


def train_epoch(model, train_loader, optimizer, criterion, device, model_type, dataset_config, scaler=None, gradient_accumulation_steps=1, epoch=0, gradient_clip=1.0):
    """训练一个epoch - 简化版，专注于稳定性和速度优化"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_periods = []
    all_probas = []
    
    # 智能梯度裁剪策略
    nan_check_interval = max(1, len(train_loader) // 20)  # 只检查5%的批次
    use_fast_clipping = gradient_clip > 2.0  # 大裁剪值使用快速模式
    nan_batch_count = 0
    max_nan_batches = min(5, len(train_loader) // 10)  # 最多跳过10%的批次
    
    # 自适应梯度裁剪 - 基于训练阶段调整
    total_epochs = getattr(dataset_config, 'epochs', 100)  # 默认总轮次
    epoch_ratio = min(epoch / total_epochs, 1.0)  # 训练进度比例
    
    # 早期训练：严格裁剪；后期训练：放宽裁剪
    if epoch_ratio < 0.3:  # 前30%轮次
        adaptive_clip = gradient_clip * 0.8  # 稍微严格
    elif epoch_ratio < 0.7:  # 中间40%轮次
        adaptive_clip = gradient_clip  # 标准值
    else:  # 后30%轮次
        adaptive_clip = gradient_clip * 1.5  # 放宽裁剪
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"训练 (clip={adaptive_clip:.1f})", ncols=120)
    accumulated_loss = 0.0
    
    for batch_idx, batch in pbar:
        try:
            # 数据处理
            if 'features' in batch:
                features = batch['features'].to(device)
                times = features[:, :, 0] if features.dim() > 2 else None
            elif 'x' in batch:
                x = batch['x'].to(device)
                features = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], 1, device=device)], dim=2)
                times = x[:, :, 0]
            else:
                continue
            
            y = batch['labels'].to(device)
            mask = batch.get('mask', torch.ones_like(features[:, :, 0])).to(device)
            periods = batch.get('periods', torch.zeros(features.shape[0])).cpu().numpy()
            
            batch_size = features.size(0)
            
            # 前向传播
            if scaler is not None:
                with autocast():
                    logits = model(features, mask)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = torch.nn.functional.cross_entropy(logits, y)
            else:
                logits = model(features, mask)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = torch.nn.functional.cross_entropy(logits, y)
            
            # NaN检查和静默跳过
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batch_count += 1
                if nan_batch_count < max_nan_batches:
                    optimizer.zero_grad()
                    continue
                else:
                    # 重置计数并继续
                    nan_batch_count = 0
                    optimizer.zero_grad()
                    continue
            else:
                nan_batch_count = 0
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    
                    # 高效NaN检查 - 只在特定批次检查
                    if batch_idx % nan_check_interval == 0 and not use_fast_clipping:
                        has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                                         for p in model.parameters() if p.grad is not None)
                        if has_nan_grad:
                            print(f"⚠️ NaN梯度检测 (batch {batch_idx}), 跳过此步")
                            optimizer.zero_grad()
                            scaler.update()
                            batch_loss = 0.0
                            continue
                    
                    # 高效梯度裁剪 - 条件执行
                    if use_fast_clipping:
                        # 快速模式：直接使用自适应裁剪值
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=adaptive_clip)
                    else:
                        # 标准模式：先计算范数再决定是否裁剪
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=adaptive_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    batch_loss = accumulated_loss * gradient_accumulation_steps
                    accumulated_loss = 0.0
                else:
                    batch_loss = loss.item() * gradient_accumulation_steps
            else:
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 非AMP路径的高效处理
                    if batch_idx % nan_check_interval == 0 and not use_fast_clipping:
                        has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                                         for p in model.parameters() if p.grad is not None)
                        if has_nan_grad:
                            print(f"⚠️ NaN梯度检测 (batch {batch_idx}), 跳过此步")
                            optimizer.zero_grad()
                            batch_loss = 0.0
                            continue
                    
                    # 高效梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=adaptive_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = accumulated_loss * gradient_accumulation_steps
                    accumulated_loss = 0.0
                else:
                    batch_loss = loss.item() * gradient_accumulation_steps
            
            # 统计
            total_loss += batch_loss
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_periods.extend(periods)
            
            probas = torch.softmax(logits, dim=1)
            all_probas.extend(probas.detach().cpu().numpy())
            
            # 更新进度条
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
        except Exception as e:
            # 静默处理所有异常
            optimizer.zero_grad()
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    # 计算实际的F1和Recall指标
    class_accuracies = {}
    additional_metrics = {}
    
    if len(all_predictions) > 0 and len(all_labels) > 0:
        try:
            # 确保数据是numpy数组
            predictions = np.array(all_predictions)
            labels = np.array(all_labels)
            
            # 计算宏平均和加权F1、Recall
            macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
            weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            
            # 计算混淆矩阵 - 零额外成本，在验证时顺便计算
            if always_compute_confusion:
                try:
                    # 获取类别数量
                    num_classes = len(np.unique(np.concatenate([labels, predictions])))
                    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
                except:
                    cm = None
            else:
                cm = None
            
            additional_metrics = {
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'macro_recall': macro_recall,
                'weighted_recall': weighted_recall,
                'f1_score': macro_f1,  # 使用宏平均，更适合不平衡数据集
                'recall': macro_recall,  # 使用宏平均，确保与accuracy不同
                'confusion_matrix': cm  # 添加混淆矩阵到返回值
            }
            
            # 计算每个类别的准确率
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_mask = (labels == label)
                if label_mask.sum() > 0:
                    class_acc = (predictions[label_mask] == labels[label_mask]).mean() * 100
                    class_accuracies[int(label)] = class_acc
                    
        except Exception:
            # 如果计算失败，使用默认值
            additional_metrics = {
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'macro_recall': 0.0,
                'weighted_recall': 0.0,
                'f1_score': 0.0,
                'recall': 0.0
            }
    else:
        additional_metrics = {
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0,
            'f1_score': 0.0,
            'recall': 0.0
        }
    
    return avg_loss, accuracy, class_accuracies, additional_metrics


def validate_epoch(model, val_loader, criterion, device, model_type, dataset_config, always_compute_confusion=True):
    """验证一个epoch - 简化版"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_probas = []
    
    pbar = tqdm(val_loader, desc="Validation", ncols=120)
    
    with torch.no_grad():
        for batch in pbar:
            try:
                # 数据处理
                if 'features' in batch:
                    features = batch['features'].to(device)
                    times = features[:, :, 0] if features.dim() > 2 else None
                elif 'x' in batch:
                    x = batch['x'].to(device)
                    features = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], 1, device=device)], dim=2)
                    times = x[:, :, 0]
                else:
                    continue
                
                y = batch['labels'].to(device)
                mask = batch.get('mask', torch.ones_like(features[:, :, 0])).to(device)
                
                # 前向传播
                logits = model(features, mask)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                loss = torch.nn.functional.cross_entropy(logits, y)
                
                # NaN检查
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
                probas = torch.softmax(logits, dim=1)
                all_probas.extend(probas.cpu().numpy())
                
                # 更新进度条
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
            except Exception:
                continue
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    # 计算实际的F1和Recall指标
    class_accuracies = {}
    additional_metrics = {}
    
    if len(all_predictions) > 0 and len(all_labels) > 0:
        try:
            # 确保数据是numpy数组
            predictions = np.array(all_predictions)
            labels = np.array(all_labels)
            
            # 计算宏平均和加权F1、Recall
            macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
            weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            
            # 计算混淆矩阵 - 零额外成本，在验证时顺便计算
            if always_compute_confusion:
                try:
                    # 获取类别数量
                    num_classes = len(np.unique(np.concatenate([labels, predictions])))
                    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
                except:
                    cm = None
            else:
                cm = None
            
            additional_metrics = {
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'macro_recall': macro_recall,
                'weighted_recall': weighted_recall,
                'f1_score': macro_f1,  # 使用宏平均，更适合不平衡数据集
                'recall': macro_recall,  # 使用宏平均，确保与accuracy不同
                'confusion_matrix': cm  # 添加混淆矩阵到返回值
            }
            
            # 计算每个类别的准确率
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_mask = (labels == label)
                if label_mask.sum() > 0:
                    class_acc = (predictions[label_mask] == labels[label_mask]).mean() * 100
                    class_accuracies[int(label)] = class_acc
                    
        except Exception:
            # 如果计算失败，使用默认值
            additional_metrics = {
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'macro_recall': 0.0,
                'weighted_recall': 0.0,
                'f1_score': 0.0,
                'recall': 0.0
            }
    else:
        additional_metrics = {
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0,
            'f1_score': 0.0,
            'recall': 0.0
        }
    
    return avg_loss, accuracy, class_accuracies, additional_metrics


def calculate_class_accuracy(predictions, labels):
    """计算各类别准确率"""
    try:
        unique_labels = np.unique(labels)
        class_accuracies = {}
        
        for label in unique_labels:
            mask = np.array(labels) == label
            class_predictions = np.array(predictions)[mask]
            class_labels = np.array(labels)[mask]
            
            if len(class_labels) > 0:
                accuracy = np.sum(class_predictions == class_labels) / len(class_labels)
                class_accuracies[int(label)] = accuracy * 100
        
        return class_accuracies
    except:
        return {}


def calculate_additional_metrics(predictions, labels, probas=None):
    """计算额外指标"""
    try:
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
        weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_recall': macro_recall,
            'weighted_recall': weighted_recall,
            'f1_score': weighted_f1,  # 兼容logging_utils的期望键名
            'recall': weighted_recall  # 兼容logging_utils的期望键名
        }
    except:
        return {
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0,
            'f1_score': 0.0,  # 兼容logging_utils的期望键名
            'recall': 0.0     # 兼容logging_utils的期望键名
        }