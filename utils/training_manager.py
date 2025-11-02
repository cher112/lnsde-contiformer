"""
训练管理器 - 整合完整的训练流程
"""

import os
import time
import torch
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
                getattr(self.args, 'gradient_accumulation_steps', 1), epoch,
                getattr(self.args, 'gradient_clip', 1.0)  # 传递梯度裁剪参数
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
            
            # 混淆矩阵已在验证时计算完成，无需重复计算
            # 如果验证时没有计算混淆矩阵，则在这里计算
            if val_metrics is None:
                val_metrics = {}
            
            # 检查验证结果是否已包含混淆矩阵
            if 'confusion_matrix' not in val_metrics or val_metrics['confusion_matrix'] is None:
                try:
                    print("⚡ 补充计算混淆矩阵...")
                    val_cm = self._calculate_confusion_matrix_only()
                    val_metrics['confusion_matrix'] = val_cm
                except Exception as e:
                    print(f"❌ 计算混淆矩阵时出错: {e}")
            else:
                print("✅ 使用验证时已计算的混淆矩阵 (零额外成本)")
            
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

    def _calculate_confusion_matrix_only(self):
        """只计算混淆矩阵，不生成图片 - 用于每个epoch的日志记录"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        # 只在验证集上计算
        with torch.no_grad():
            for batch in self.val_loader:
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
                    outputs = self.model(features, mask)
                else:
                    outputs = self.model(features)
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 转换为numpy数组并计算混淆矩阵
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # 获取类别数量
        class_names = self._get_class_names()
        num_classes = len(class_names)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        return cm
        
        
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
                        outputs = self.model(features, mask)
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
        
        # 生成混淆矩阵 - 适配正确的类别数量
        num_classes = len(class_names)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        # 配置中文字体
        self._configure_chinese_font()
        
        # 创建双混淆矩阵可视化 - 原始数量 + 归一化百分比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：原始数量混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'label': '样本数量'})
        
        ax1.set_title(f'{self.args.dataset_name} 原始混淆矩阵\n'
                     f'模型: {self.args.model_type} | 总准确率: {np.trace(cm)/np.sum(cm):.3f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('预测类别')
        ax1.set_ylabel('真实类别')
        
        # 右图：归一化百分比混淆矩阵
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = np.nan_to_num(cm_percent, nan=0.0)  # 处理除零情况
        
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=ax2, cbar_kws={'label': '预测百分比 (%)'})
        
        ax2.set_title(f'{self.args.dataset_name} 归一化混淆矩阵\n'
                     f'按行归一化 (召回率视角)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测类别')
        ax2.set_ylabel('真实类别')
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存到pics目录
        pics_dir = f"/root/autodl-tmp/lnsde-contiformer/results/pics/{self.args.dataset_name}"
        os.makedirs(pics_dir, exist_ok=True)
        
        model_name_parts = []
        if self.args.use_sde:
            model_name_parts.append("SDE")
        if self.args.use_contiformer:
            model_name_parts.append("ContiFormer")
        if self.args.use_cga:
            model_name_parts.append("CGA")
        model_name = "_".join(model_name_parts) if model_name_parts else "Base"
        
        confusion_matrix_path = os.path.join(pics_dir, f"{self.args.dataset_name.lower()}_dual_confusion_matrix_{model_name.lower()}.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 混淆矩阵已保存: {confusion_matrix_path}")
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_path = os.path.join(pics_dir, f"{self.args.dataset_name.lower()}_classification_report_{model_name.lower()}.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"{self.args.dataset_name} 分类报告\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"SDE: {'On' if self.args.use_sde else 'Off'}\n")
            f.write(f"ContiFormer: {'On' if self.args.use_contiformer else 'Off'}\n")
            f.write(f"CGA: {'On' if self.args.use_cga else 'Off'}\n\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names))
            
        print(f"✅ 分类报告已保存: {report_path}")
        accuracy = accuracy_score(y_true, y_pred)
        print(f"总体准确率: {accuracy:.4f}")
        print(f"加权F1分数: {report['weighted avg']['f1-score']:.4f}")
        print(f"宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
        
        # 返回混淆矩阵供日志记录使用
        return cm
    
    def _get_class_names(self):
        """获取类别名称 - 根据数据集适配5-5-7分类"""
        if self.args.dataset_name == "ASAS":
            # ASAS: 5个类别
            return ["Beta_Persei", "Delta_Scuti", "RR_Lyrae_FM", "RR_Lyrae_FO", "W_Ursae_Maj"]
        elif self.args.dataset_name == "LINEAR": 
            # LINEAR: 5个类别
            return ["Beta_Persei", "Delta_Scuti", "RR_Lyrae_FM", "RR_Lyrae_FO", "W_Ursae_Maj"]
        elif self.args.dataset_name == "MACHO":
            # MACHO: 7个类别
            return ["Be", "CEPH", "EB", "LPV", "MOA", "QSO", "RRL"]
        else:
            # 默认类别名称
            num_classes = getattr(self.model, 'num_classes', 7)
            return [f"Class_{i}" for i in range(num_classes)]
    
    def _configure_chinese_font(self):
        """配置中文字体显示 - 解决Font 'default'问题"""
        import matplotlib.font_manager as fm
        
        # 添加字体到matplotlib管理器
        try:
            fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
            fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
        except:
            pass
        
        # 设置中文字体优先级列表
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        
        # 清理matplotlib缓存并刷新字体
        try:
            # 清理matplotlib缓存
            import shutil
            cache_dir = fm.get_cachedir()
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            fm._rebuild()
        except:
            # 备选方法：重新加载字体管理器
            fm.fontManager.__init__()
        
        print("✓ 中文字体配置成功: WenQuanYi Zen Hei")
