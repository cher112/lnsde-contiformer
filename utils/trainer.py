"""
Training utilities
"""

import torch
import numpy as np
from typing import Dict, Any


class SDEContiformerTrainer:
    """SDE ContiFormer训练器（简化版）"""
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer  
        self.criterion = criterion
        
    def train_step(self, batch):
        """训练步骤"""
        self.model.train()
        # 实现细节在main.py中
        pass


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


class ModelCheckpoint:
    """模型检查点"""
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = None
        
    def __call__(self, model, current_score):
        if not self.save_best_only or self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)
            return True
        return False