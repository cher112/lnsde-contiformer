#!/usr/bin/env python3
"""
训练进度监控脚本
监控所有数据集在不同时间间隔下的训练状态
"""

import os
import time
import subprocess
import threading
from datetime import datetime

class TrainingMonitor:
    def __init__(self, base_dir="/root/autodl-tmp/lnsde-contiformer"):
        self.base_dir = base_dir
        self.training_configs = [
            {"dataset": 1, "interval": 0.005, "name": "ASAS_005", "status": "running"},
            {"dataset": 1, "interval": 0.02, "name": "ASAS_02", "status": "pending"},
            {"dataset": 2, "interval": 0.005, "name": "LINEAR_005", "status": "pending"},
            {"dataset": 2, "interval": 0.02, "name": "LINEAR_02", "status": "pending"}, 
            {"dataset": 3, "interval": 0.005, "name": "MACHO_005", "status": "pending"},
            {"dataset": 3, "interval": 0.02, "name": "MACHO_02", "status": "pending"}
        ]
        self.completed_trainings = []
        
    def check_training_status(self):
        """检查训练状态"""
        # 检查screen会话
        try:
            result = subprocess.run(['screen', '-list'], capture_output=True, text=True)
            screen_sessions = result.stdout
            
            for config in self.training_configs:
                if config["status"] == "running":
                    session_name = f"{config['name'].lower()}"
                    if session_name not in screen_sessions.lower():
                        print(f"⚠️  {config['name']} 训练会话已结束")
                        config["status"] = "completed"
                        self.completed_trainings.append(config)
                        
        except Exception as e:
            print(f"检查screen会话出错: {e}")
            
    def start_next_training(self):
        """启动下一个训练"""
        for config in self.training_configs:
            if config["status"] == "pending":
                print(f"🚀 启动 {config['name']} 训练...")
                
                # 构建训练命令
                cmd = (f"cd {self.base_dir} && "
                      f"python main.py --dataset {config['dataset']} "
                      f"--min_time_interval {config['interval']} "
                      f"--model_type 1 --epochs 50 --batch_size 32 "
                      f"--learning_rate 1e-4 --load_model 0 "
                      f"> {config['name'].lower()}.log 2>&1")
                
                # 启动screen会话
                screen_cmd = f"screen -dmS {config['name'].lower()} bash -c \"{cmd}\""
                
                try:
                    subprocess.run(screen_cmd, shell=True, check=True)
                    config["status"] = "running"
                    print(f"✅ {config['name']} 训练已启动")
                    return True
                except Exception as e:
                    print(f"❌ 启动 {config['name']} 失败: {e}")
                    return False
        
        return False
        
    def get_training_progress(self, log_file):
        """获取训练进度"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 查找最新的epoch信息
                for line in reversed(lines[-50:]):  # 只看最后50行
                    if "Epoch [" in line and "/50]" in line:
                        return line.strip()
                    elif "训练完成" in line:
                        return "训练已完成"
                        
        except Exception as e:
            return f"无法读取日志: {e}"
            
        return "等待训练开始..."
        
    def monitor_loop(self):
        """监控循环"""
        print("🔍 开始监控训练进度...")
        
        while True:
            os.system('clear')
            print(f"{'='*60}")
            print(f"训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            self.check_training_status()
            
            # 显示所有训练状态
            running_count = 0
            for config in self.training_configs:
                status_emoji = {
                    "running": "🏃", 
                    "completed": "✅", 
                    "pending": "⏳",
                    "failed": "❌"
                }.get(config["status"], "❓")
                
                log_file = f"{self.base_dir}/{config['name'].lower()}.log"
                progress = self.get_training_progress(log_file)
                
                print(f"{status_emoji} {config['name']:<12} | {config['status']:<10} | {progress}")
                
                if config["status"] == "running":
                    running_count += 1
                    
            print(f"{'='*60}")
            print(f"运行中: {running_count} | 已完成: {len(self.completed_trainings)} | 总数: {len(self.training_configs)}")
            
            # 如果没有训练在运行，启动下一个
            if running_count == 0:
                if not self.start_next_training():
                    print("🎉 所有训练任务已完成！")
                    break
            
            time.sleep(30)  # 30秒检查一次
            
    def integrate_logs(self):
        """整合日志文件到docs目录"""
        docs_dir = f"{self.base_dir}/docs/training_logs"
        os.makedirs(docs_dir, exist_ok=True)
        
        print("📁 整合日志文件到docs目录...")
        
        # 复制根目录下的日志文件
        for config in self.training_configs:
            log_file = f"{self.base_dir}/{config['name'].lower()}.log"
            if os.path.exists(log_file):
                target_file = f"{docs_dir}/{config['name']}_interval_{str(config['interval']).replace('.', '_')}.log"
                try:
                    subprocess.run(f"cp {log_file} {target_file}", shell=True, check=True)
                    print(f"✅ 已复制: {config['name']}")
                except Exception as e:
                    print(f"❌ 复制失败 {config['name']}: {e}")
        
        # 复制results目录下的日志文件
        results_dir = f"{self.base_dir}/results"
        if os.path.exists(results_dir):
            try:
                subprocess.run(f"find {results_dir} -name '*.log' -exec cp {{}} {docs_dir}/ \\;", shell=True)
                print("✅ 已复制results目录下的所有日志文件")
            except Exception as e:
                print(f"❌ 复制results日志失败: {e}")
        
        print(f"📂 日志文件已整合到: {docs_dir}")

if __name__ == "__main__":
    import sys
    
    monitor = TrainingMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "integrate":
        monitor.integrate_logs()
    else:
        try:
            monitor.monitor_loop()
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
            monitor.integrate_logs()