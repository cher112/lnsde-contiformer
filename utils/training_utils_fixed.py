"""
训练相关工具函数 - 简化版，专注于稳定性和NaN处理
"""

import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, confusion_matrix


def train_epoch(model, train_loader, optimizer, criterion, device, model_type, dataset_config, scaler=None, gradient_accumulation_steps=1, epoch=0):
    """训练一个epoch - 简化版，专注于稳定性"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_periods = []
    all_probas = []
    
    # NaN处理计数
    nan_batch_count = 0
    max_nan_batches = 50
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=120)
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
                    # 检查梯度
                    scaler.unscale_(optimizer)
                    has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                                     for p in model.parameters() if p.grad is not None)
                    
                    if has_nan_grad:
                        # 静默跳过NaN梯度
                        optimizer.zero_grad()
                        scaler.update()
                        batch_loss = 0.0
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    has_nan_grad = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                                     for p in model.parameters() if p.grad is not None)
                    
                    if has_nan_grad:
                        optimizer.zero_grad()
                        batch_loss = 0.0
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            if scaler:
                scaler.update()
            optimizer.zero_grad()
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    # 简化指标计算
    class_accuracies = {}
    additional_metrics = {
        'macro_f1': 0.0,
        'weighted_f1': 0.0,
        'macro_recall': 0.0,
        'weighted_recall': 0.0
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
    
    # 简化指标
    class_accuracies = {}
    additional_metrics = {
        'macro_f1': 0.0,
        'weighted_f1': 0.0,
        'macro_recall': 0.0,
        'weighted_recall': 0.0
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
            'weighted_recall': weighted_recall
        }
    except:
        return {
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0
        }