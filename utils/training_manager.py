"""
训练管理器 - 整合完整的训练流程
"""

import os
import time
import torch
from torch.cuda.amp import GradScaler

from .system_utils import clear_gpu_memory
from .model_utils import save_checkpoint, setup_model_save_paths
from .logging_utils import update_log, print_epoch_summary
from .training_utils import train_epoch, validate_epoch
from .visualization import generate_training_visualizations


class TrainingManager:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, args, dataset_config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.args = args
        self.dataset_config = dataset_config
        self.scheduler = scheduler
        
        # 混合精度训练
        self.use_amp = hasattr(args, 'use_amp') and args.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # 设置模型保存路径
        self.model_save_dir = setup_model_save_paths(args.save_dir, args.dataset_name, args.model_type)
        
    def run_training(self, log_path, log_data, best_val_acc=0.0, start_epoch=0):
        """运行完整的训练流程"""
        print("=== 开始训练 ===")
        
        if start_epoch > 0:
            print(f"从第 {start_epoch + 1} 轮继续训练...")
        else:
            print("从头开始训练...")
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch [{epoch + 1}/{self.args.epochs}]")
            
            # 训练阶段
            train_loss, train_acc, train_class_acc, train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion, 
                self.device, self.args.model_type, self.dataset_config, self.scaler
            )
            
            # 验证阶段
            val_loss, val_acc, val_class_acc, val_metrics = validate_epoch(
                self.model, self.val_loader, self.criterion, 
                self.device, self.args.model_type, self.dataset_config
            )
            
            # 检查是否是最佳模型
            is_best = val_acc > best_val_acc
            
            # 如果是最佳模型，重新验证以获取混淆矩阵
            if is_best:
                print("🎯 检测到最佳模型，重新验证以获取混淆矩阵...")
                _, _, _, val_metrics_with_cm = validate_epoch(
                    self.model, self.val_loader, self.criterion, 
                    self.device, self.args.model_type, self.dataset_config, 
                    compute_confusion=True
                )
                # 更新验证指标以包含混淆矩阵
                val_metrics = val_metrics_with_cm
            
            epoch_time = time.time() - epoch_start_time
            
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印轮次总结
            print_epoch_summary(
                epoch + 1, self.args.epochs, train_loss, train_acc,
                val_loss, val_acc, val_class_acc if is_best else None, 
                epoch_time, current_lr, train_metrics, val_metrics
            )
            
            # 更新日志
            update_log(
                log_path, log_data, epoch + 1, train_loss, train_acc,
                val_loss, val_acc, val_class_acc, epoch_time, current_lr, 
                train_metrics, val_metrics, is_best
            )
            
            # 保存检查点
            self._save_checkpoints(epoch, val_loss, val_acc, best_val_acc)
            
            # 更新最佳验证准确率
            if is_best:
                best_val_acc = val_acc
                print(f"🎉 新的最佳验证准确率: {best_val_acc:.4f}%")
            
            # 清理GPU内存
            clear_gpu_memory()
        
        print(f"\n=== 训练完成 ===")
        print(f"最佳验证准确率: {best_val_acc:.4f}%")
        
        # 生成训练可视化图表
        print("=== 生成训练可视化图表 ===")
        try:
            # 从args获取模型类型字符串
            model_type_map = {1: 'langevin', 2: 'linear_noise', 3: 'geometric'}
            model_type_str = model_type_map.get(self.args.model_type, 'unknown')
            
            # 从log_path获取日期信息
            import os
            from datetime import datetime
            log_dir_parts = log_path.split(os.sep)
            date_str = None
            for part in log_dir_parts:
                if len(part) == 8 and part.isdigit():  # 格式如 20250826
                    date_str = part
                    break
            
            if date_str is None:
                date_str = datetime.now().strftime('%Y%m%d')
            
            generated_files = generate_training_visualizations(
                log_path, self.args.dataset_name, model_type_str, date_str
            )
            
            if generated_files:
                print(f"✅ 成功生成 {len(generated_files)} 个可视化图表")
            else:
                print("⚠️  可视化图表生成失败")
                
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
        
        return best_val_acc
    
    def _save_checkpoints(self, epoch, val_loss, val_acc, best_val_acc):
        """保存模型检查点"""
        # 定期保存epoch模型
        if (epoch + 1) % self.args.save_interval == 0:
            epoch_save_path = os.path.join(
                self.model_save_dir,
                f"{self.args.dataset_name}_{self.args.model_type}_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(self.model, self.optimizer, epoch, val_loss, val_acc, epoch_save_path)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_save_path = os.path.join(
                self.model_save_dir,
                f"{self.args.dataset_name}_{self.args.model_type}_best.pth"
            )
            
            # 添加模型参数到检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'best_val_accuracy': val_acc,
                'model_params': {
                    'hidden_channels': self.args.hidden_channels,
                    'contiformer_dim': self.args.contiformer_dim,
                    'n_heads': self.args.n_heads,
                    'n_layers': self.args.n_layers,
                    'sde_method': self.args.sde_method,
                    'dt': self.args.dt,
                    'rtol': self.args.rtol,
                    'atol': self.args.atol,
                }
            }
            
            torch.save(checkpoint, best_save_path)
            print(f"✅ 保存最佳模型: {best_save_path}")