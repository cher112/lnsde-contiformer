"""
训练相关工具函数
"""

import torch
import numpy as np
from torch.cuda.amp import autocast
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, confusion_matrix


def train_epoch(model, train_loader, optimizer, criterion, device, model_type, dataset_config, scaler=None):
    """训练一个epoch - 支持混合精度训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_periods = []
    all_probas = []
    
    # 创建进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", 
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, batch in pbar:
        optimizer.zero_grad()
        
        # 数据移动到设备 - 转换为3维features格式
        x = batch['x'].to(device)  # (batch, seq_len, 2) [time, mag]
        y = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        
        # 将2维x转换为3维features [time, mag, errmag] 
        batch_size, seq_len = x.shape[0], x.shape[1]
        errmag = torch.zeros(batch_size, seq_len, 1, device=device)
        features = torch.cat([x, errmag], dim=2)  # (batch, seq_len, 3)
        
        # 正确提取时间数据：从x的第一个维度
        times = x[:, :, 0]  # (batch, seq_len) 真实的时间序列
        
        periods = batch['periods'].cpu().numpy()
        
        batch_size = x.size(0)
        
        # 混合精度训练
        if scaler is not None:
            with autocast():
                if model_type == 'geometric':
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )[0]
                elif model_type == 'langevin':
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]  # 取总损失
                else:  # linear_noise
                    logits, sde_features = model(features, mask)
                    loss = model.compute_loss(
                        logits, y, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]  # 取总损失
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            if model_type == 'geometric':
                logits, sde_features = model(features, times, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )[0]
            elif model_type == 'langevin':
                logits, sde_features = model(features, times, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]  # 取总损失
            else:  # linear_noise
                logits, sde_features = model(features, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]  # 取总损失
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
        # 保存预测和标签用于后续分析
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_periods.extend(periods)
        
        # 保存概率用于计算AUROC
        probas = torch.softmax(logits, dim=1)
        all_probas.extend(probas.detach().cpu().numpy())
        
        # 计算当前指标（每10个batch更新一次）
        current_acc = 100. * correct / total
        postfix = {'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'}
        
        # 每10个batch计算并显示额外指标
        if (batch_idx + 1) % 10 == 0 and len(all_predictions) >= 10:
            batch_metrics = calculate_additional_metrics(
                all_predictions, all_labels, np.array(all_probas) if all_probas else None
            )
            postfix.update({
                'F1': f'{batch_metrics["f1_score"]:.1f}',
                'Rec': f'{batch_metrics["recall"]:.1f}'
            })
        
        pbar.set_postfix(postfix)
    
    pbar.close()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # 计算各类别准确率
    class_accuracies = calculate_class_accuracy(all_predictions, all_labels)
    
    # 计算额外指标
    additional_metrics = calculate_additional_metrics(
        all_predictions, all_labels, np.array(all_probas) if all_probas else None
    )
    
    return avg_loss, accuracy, class_accuracies, additional_metrics


def validate_epoch(model, val_loader, criterion, device, model_type, dataset_config, always_compute_confusion=True):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_periods = []
    all_probas = []
    
    # 创建进度条
    pbar = tqdm(val_loader, desc="Validation", 
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        for batch in pbar:
            # 数据移动到设备
            x = batch['x'].to(device)  # (batch, seq_len, 2) [time, mag] 
            y = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            
            # 将2维x转换为3维features [time, mag, errmag]
            batch_size, seq_len = x.shape[0], x.shape[1]
            errmag = torch.zeros(batch_size, seq_len, 1, device=device)
            features = torch.cat([x, errmag], dim=2)  # (batch, seq_len, 3)
            
            # 正确提取时间数据：从x的第一个维度
            times = x[:, :, 0]  # (batch, seq_len) 真实的时间序列
            
            periods = batch['periods'].cpu().numpy()
            
            # 前向传播
            if model_type == 'geometric':
                logits, sde_features = model(features, times, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )[0]
            elif model_type == 'langevin':
                logits, sde_features = model(features, times, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]  # 取总损失
            else:  # linear_noise
                logits, sde_features = model(features, mask)
                loss = model.compute_loss(
                    logits, y, sde_features,
                    focal_gamma=dataset_config['focal_gamma'],
                    temperature=dataset_config['temperature']
                )
                if isinstance(loss, tuple):
                    loss = loss[0]
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # 保存预测和标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_periods.extend(periods)
            
            # 保存概率用于计算AUROC
            probas = torch.softmax(logits, dim=1)
            all_probas.extend(probas.detach().cpu().numpy())
            
            # 更新进度条
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    pbar.close()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # 计算各类别准确率
    class_accuracies = calculate_class_accuracy(all_predictions, all_labels)
    
    # 计算额外指标 - 默认总是计算混淆矩阵
    additional_metrics = calculate_additional_metrics(
        all_predictions, all_labels, np.array(all_probas) if all_probas else None, compute_confusion=always_compute_confusion
    )
    
    return avg_loss, accuracy, class_accuracies, additional_metrics


def calculate_class_accuracy(predictions, labels):
    """计算各类别的准确率"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 获取所有类别
    unique_labels = np.unique(labels)
    class_accuracies = {}
    
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            class_predictions = predictions[mask]
            class_labels = labels[mask]
            accuracy = (class_predictions == class_labels).mean() * 100
            class_accuracies[f'class_{label}'] = accuracy
    
    return class_accuracies


def calculate_additional_metrics(predictions, labels, probas=None, compute_confusion=False):
    """计算额外的评价指标：F1, Recall，可选混淆矩阵"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {}
    
    try:
        # F1 Score (macro平均)
        f1_macro = f1_score(labels, predictions, average='macro')
        metrics['f1_score'] = f1_macro * 100  # 转为百分比
        
        # Recall (macro平均)
        recall_macro = recall_score(labels, predictions, average='macro')
        metrics['recall'] = recall_macro * 100  # 转为百分比
        
        # 混淆矩阵（仅在需要时计算）
        if compute_confusion:
            cm = confusion_matrix(labels, predictions)
            metrics['confusion_matrix'] = cm.tolist()  # 转为列表便于JSON序列化
            
    except Exception as e:
        # 如果计算失败，设置默认值
        metrics['f1_score'] = 0.0
        metrics['recall'] = 0.0
        if compute_confusion:
            metrics['confusion_matrix'] = []
        print(f"Warning: Failed to calculate metrics: {e}")
    
    return metrics