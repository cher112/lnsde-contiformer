"""
训练管理器 - 整合完整的训练流程
"""

import os
import time
import torch
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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
        
        # 使用标准化路径 - models 子目录已在 timestamp_path 中创建
        self.model_save_dir = os.path.join(args.save_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
    def run_training(self, log_path, log_data, best_val_acc=0.0, start_epoch=0):
        """运行完整的训练流程"""
        print("=== 开始训练 ===")
        
        if start_epoch > 0:
            print(f"从第 {start_epoch} 轮继续训练...")
        else:
            print("从头开始训练...")
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch [{epoch + 1}/{self.args.epochs}]")
            
            # 训练阶段
            train_loss, train_acc, train_class_acc, train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion, 
                self.device, self.args.model_type, self.dataset_config, self.scaler,
                getattr(self.args, 'gradient_accumulation_steps', 1)
            )
            
            # 验证阶段 - 总是计算混淆矩阵
            val_loss, val_acc, val_class_acc, val_metrics = validate_epoch(
                self.model, self.val_loader, self.criterion, 
                self.device, self.args.model_type, self.dataset_config
            )
            
            # 检查是否是最佳模型
            is_best = val_acc > best_val_acc
            
            epoch_time = time.time() - epoch_start_time
            
            # 更新学习率调度器 - ReduceLROnPlateau现在监控验证准确率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)  # 传入验证准确率
                else:
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
        
        # 移除多余输出
        # print(f"\n=== 训练完成 ===")
        print(f"最佳验证准确率: {best_val_acc:.4f}%")
        
        # 生成训练可视化图表
        print("=== 生成训练可视化图表 ===")
        try:
            # 从args获取模型类型字符串
            model_type_map = {1: 'langevin', 2: 'linear_noise', 3: 'geometric'}
            model_type_str = model_type_map.get(self.args.model_type, 'unknown')
            
            # 使用标准化路径 - timestamp_dir 就是 args.save_dir
            timestamp_dir = self.args.save_dir
            
            generated_files = generate_training_visualizations(
                log_path, self.args.dataset_name, model_type_str, timestamp_dir
            )
            
            if generated_files:
                print(f"✅ 成功生成 {len(generated_files)} 个可视化图表")
            else:
                print("⚠️  可视化图表生成失败")
                
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
        
        # 生成完整数据集的混淆矩阵
        print("=== 生成混淆矩阵 ===")
        try:
            self._generate_confusion_matrix()
        except Exception as e:
            print(f"❌ 生成混淆矩阵时出错: {e}")
        
        return best_val_acc
    
    def _save_checkpoints(self, epoch, val_loss, val_acc, best_val_acc):
        """保存模型检查点 - 使用详细参数命名"""
        # 生成详细的模型名称
        model_name_parts = [
            self.args.dataset_name,
            self.args.model_type,
            f"sde{'1' if self.args.use_sde else '0'}",
            f"cf{'1' if self.args.use_contiformer else '0'}", 
            f"cga{'1' if self.args.use_cga else '0'}",
            f"hc{self.args.hidden_channels}",
            f"cd{self.args.contiformer_dim}",
            f"lr{self.args.learning_rate:.0e}",
            f"bs{self.args.batch_size}"
        ]
        base_model_name = "_".join(model_name_parts)
        
        # 定期保存epoch模型
        if (epoch + 1) % self.args.save_interval == 0:
            epoch_save_path = os.path.join(
                self.model_save_dir,
                f"{base_model_name}_epoch{epoch + 1}.pth"
            )
            save_checkpoint(self.model, self.optimizer, epoch, val_loss, val_acc, epoch_save_path)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_save_path = os.path.join(
                self.model_save_dir,
                f"{base_model_name}_best.pth"
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
                    'use_sde': self.args.use_sde,
                    'use_contiformer': self.args.use_contiformer,
                    'use_cga': self.args.use_cga
                },
                'training_params': {
                    'learning_rate': self.args.learning_rate,
                    'batch_size': self.args.batch_size,
                    'epochs': self.args.epochs,
                    'dataset': self.args.dataset_name
                }
            }
            
            torch.save(checkpoint, best_save_path)
            print(f"✅ 保存最佳模型: {best_save_path}")    
    def _generate_confusion_matrix(self):
        """生成完整数据集的混淆矩阵"""
        print("正在生成混淆矩阵...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        # 在完整数据集（训练+验证）上评估
        datasets = [("Train", self.train_loader), ("Val", self.val_loader)]
        
        with torch.no_grad():
            for dataset_name, dataloader in datasets:
                for batch in dataloader:
                    if isinstance(batch, dict):
                        features = batch["features"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        mask = batch.get("mask", None)
                        if mask is not None:
                            mask = mask.to(self.device)
                    else:
                        features, labels = batch
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        mask = None
                    
                    # 前向传播
                    if self.args.model_type in ["linear_noise", "langevin", "geometric"]:
                        outputs, _ = self.model(features, mask)
                    else:
                        outputs = self.model(features)
                    
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # 转换为numpy数组
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # 获取类别名称
        class_names = self._get_class_names()
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 配置中文字体
        self._configure_chinese_font()
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{self.args.dataset_name} - 混淆矩阵\n"
                 f"模型: {self.args.model_type} | "
                 f"SDE: {'On' if self.args.use_sde else 'Off'} | "
                 f"ContiFormer: {'On' if self.args.use_contiformer else 'Off'} | "
                 f"CGA: {'On' if self.args.use_cga else 'Off'}")
        plt.xlabel("预测类别")
        plt.ylabel("真实类别")
        
        # 保存混淆矩阵
        model_name_parts = [
            self.args.dataset_name,
            self.args.model_type,
            f"sde{'1' if self.args.use_sde else '0'}",
            f"cf{'1' if self.args.use_contiformer else '0'}", 
            f"cga{'1' if self.args.use_cga else '0'}",
            f"hc{self.args.hidden_channels}",
            f"cd{self.args.contiformer_dim}"
        ]
        base_name = "_".join(model_name_parts)
        
        matrix_filename = f"{base_name}_confusion_matrix.png"
        matrix_path = os.path.join(self.args.save_dir, matrix_filename)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 混淆矩阵已保存: {matrix_path}")
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_filename = f"{base_name}_classification_report.txt"
        report_path = os.path.join(self.args.save_dir, report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"{self.args.dataset_name} 分类报告\n")
            f.write(f"模型: {self.args.model_type}\n")
            f.write(f"SDE: {'On' if self.args.use_sde else 'Off'}\n")
            f.write(f"ContiFormer: {'On' if self.args.use_contiformer else 'Off'}\n")
            f.write(f"CGA: {'On' if self.args.use_cga else 'Off'}\n\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names))
        
        print(f"✅ 分类报告已保存: {report_path}")
        
        # 计算总体指标
        accuracy = (y_true == y_pred).mean()
        print(f"总体准确率: {accuracy:.4f}")
        print(f"加权F1分数: {report['weighted avg']['f1-score']:.4f}")
        print(f"宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
    
    def _get_class_names(self):
        """获取类别名称"""
        if self.args.dataset_name == "ASAS":
            return ["Beta_Persei", "RR_Lyrae_FM", "W_Ursae_Maj"]
        elif self.args.dataset_name == "LINEAR":
            return ["Beta_Persei", "Delta_Scuti", "RR_Lyrae_FM", "RR_Lyrae_FO", "W_Ursae_Maj"]
        elif self.args.dataset_name == "MACHO":
            return ["RR_Lyrae_ab", "RR_Lyrae_c", "Cepheid", "EB", "Long_Period_Variable", "Non_Variable", "Quasar"]
        else:
            # 默认类别名称
            return [f"Class_{i}" for i in range(self.model.num_classes)]
    
    def _configure_chinese_font(self):
        """配置中文字体显示"""
        import matplotlib.font_manager as fm
        
        # 添加字体到matplotlib管理器
        try:
            fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
            fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
        except:
            pass
        
        # 设置中文字体优先级列表
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
