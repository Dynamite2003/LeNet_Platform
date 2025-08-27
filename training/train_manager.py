import threading
import time
import uuid
import subprocess
import sys
from datetime import datetime
import json
import os
import re
from queue import Queue, Empty
from django.utils import timezone
from .models import TrainingRecord

class TrainingManager:
    """训练管理器 - 管理所有训练任务的状态和进度"""
    
    def __init__(self):
        self.active_trainings = {}  # 存储活跃的训练任务（内存中）
        
    def start_training(self, config):
        """开始训练任务"""
        training_id = str(uuid.uuid4())
        
        # 创建数据库记录
        training_record = TrainingRecord.objects.create(
            training_id=training_id,
            status='preparing',
            config=config,
            started_at=timezone.now(),
            total_epochs=config.get('epochs', 10),
        )
        
        # 创建内存中的状态（用于实时更新）
        training_state = {
            'id': training_id,
            'status': 'preparing',
            'config': config,
            'start_time': timezone.now().isoformat(),
            'current_epoch': 0,
            'total_epochs': config.get('epochs', 10),
            'current_loss': 0.0,
            'current_accuracy': 0.0,
            'current_val_loss': 0.0,
            'epoch_losses': [],
            'epoch_accuracies': [],
            'epoch_val_losses': [],
            'step_losses': [],
            'step_labels': [],
            'logs': [],
            'progress': 0.0,
            'error_message': None,
            'model_path': None,
            'total_samples': 0,
            'current_samples': 0,
            'samples_per_epoch': 0,
            'error_samples': []
        }
        
        self.active_trainings[training_id] = training_state
        
        # 在新线程中启动训练
        training_thread = threading.Thread(
            target=self._run_training,
            args=(training_id, config),
            daemon=True
        )
        training_thread.start()
        
        return training_id
    
    def get_training_status(self, training_id):
        """获取训练状态"""
        # 首先检查内存中的活跃训练
        if training_id in self.active_trainings:
            return self.active_trainings[training_id]
        
        # 如果不在内存中，从数据库获取
        try:
            training_record = TrainingRecord.objects.get(training_id=training_id)
            return training_record.to_dict()
        except TrainingRecord.DoesNotExist:
            return None
    
    def stop_training(self, training_id):
        """停止训练任务"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id]['status'] = 'stopping'
            
            # 更新数据库状态
            try:
                training_record = TrainingRecord.objects.get(training_id=training_id)
                training_record.status = 'stopping'
                training_record.save()
            except TrainingRecord.DoesNotExist:
                pass
            
            # 如果有进程在运行，尝试终止它
            if 'process' in self.active_trainings[training_id]:
                process = self.active_trainings[training_id]['process']
                if process.poll() is None:  # 进程仍在运行
                    try:
                        process.terminate()
                        # 等待一会儿，如果还没结束就强制kill
                        import time
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                    except Exception as e:
                        self._log_message(training_id, f"停止进程时出错: {str(e)}")
            
            return True
        return False
    
    def get_all_trainings(self):
        """获取所有训练任务"""
        # 获取活跃的训练任务
        active = list(self.active_trainings.values())
        
        # 从数据库获取所有历史训练记录
        all_records = TrainingRecord.objects.all().order_by('-created_at')
        history = []
        
        # 将数据库记录转换为字典格式，排除已经在active中的
        active_ids = {t['id'] for t in active}
        for record in all_records:
            if record.training_id not in active_ids:
                history.append(record.to_dict())
        
        return {
            'active': active,
            'history': history
        }
    
    def get_all_training_records(self):
        """获取所有训练记录（包括活跃和历史）"""
        # 合并活跃训练和数据库记录
        all_trainings = []
        
        # 添加活跃训练
        for training in self.active_trainings.values():
            all_trainings.append(training)
        
        # 添加数据库中的历史记录（排除已经在活跃训练中的）
        active_ids = set(self.active_trainings.keys())
        for record in TrainingRecord.objects.all():
            if record.training_id not in active_ids:
                all_trainings.append(record.to_dict())
        
        # 按开始时间排序（最新的在前）
        all_trainings.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return all_trainings
    
    def _log_message(self, training_id, message):
        """添加日志消息"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        
        # 更新内存状态
        if training_id in self.active_trainings:
            self.active_trainings[training_id]['logs'].append(log_entry)
        
        # 更新数据库
        try:
            training_record = TrainingRecord.objects.get(training_id=training_id)
            logs = training_record.logs
            logs.append(log_entry)
            training_record.logs = logs
            training_record.save(update_fields=['logs'])
        except TrainingRecord.DoesNotExist:
            pass
    
    def _update_progress(self, training_id, epoch, total_epochs, loss=None, accuracy=None):
        """更新训练进度"""
        # 更新内存状态
        if training_id in self.active_trainings:
            state = self.active_trainings[training_id]
            state['current_epoch'] = epoch
            state['progress'] = (epoch / total_epochs) * 100
            
            if loss is not None:
                state['current_loss'] = loss
                state['epoch_losses'].append(loss)
            
            if accuracy is not None:
                state['current_accuracy'] = accuracy
                state['epoch_accuracies'].append(accuracy)
        
        # 更新数据库
        self._sync_to_database(training_id)
    
    def _sync_to_database(self, training_id):
        """将内存状态同步到数据库"""
        if training_id not in self.active_trainings:
            return
        
        try:
            training_record = TrainingRecord.objects.get(training_id=training_id)
            state = self.active_trainings[training_id]
            
            # 更新所有字段
            training_record.status = state['status']
            training_record.current_epoch = state['current_epoch']
            training_record.progress = state['progress']
            training_record.current_loss = state['current_loss']
            training_record.current_accuracy = state['current_accuracy']
            training_record.current_val_loss = state['current_val_loss']
            training_record.epoch_losses = state['epoch_losses']
            training_record.epoch_accuracies = state['epoch_accuracies']
            training_record.epoch_val_losses = state['epoch_val_losses']
            training_record.step_losses = state['step_losses']
            training_record.step_labels = state['step_labels']
            training_record.logs = state['logs']
            training_record.total_samples = state['total_samples']
            training_record.current_samples = state['current_samples']
            training_record.samples_per_epoch = state['samples_per_epoch']
            training_record.error_samples = state['error_samples']
            training_record.error_message = state['error_message']
            training_record.model_path = state['model_path']
            
            training_record.save()
        except TrainingRecord.DoesNotExist:
            pass
    
    def _run_training(self, training_id, config):
        """在后台线程中运行训练 - 直接调用train.py"""
        try:
            self._log_message(training_id, "开始准备训练...")
            self.active_trainings[training_id]['status'] = 'running'
            
            # 更新数据库状态
            try:
                training_record = TrainingRecord.objects.get(training_id=training_id)
                training_record.status = 'running'
                training_record.save()
            except TrainingRecord.DoesNotExist:
                pass
            
            # 准备训练参数
            epochs = config.get('epochs', 10)
            batch_size = config.get('batch_size', 64)
            learning_rate = config.get('learning_rate', 0.001)
            optimizer = config.get('optimizer', 'sgd')
            
            # 优化器参数
            momentum = config.get('momentum', 0.9)
            weight_decay = config.get('weight_decay', 1e-4)
            beta1 = config.get('beta1', 0.9)
            beta2 = config.get('beta2', 0.999)
            eps = config.get('eps', 1e-8)
            
            self._log_message(training_id, f"训练配置: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, optimizer={optimizer}")
            
            # 构建train.py命令，添加-u参数强制不缓冲输出
            cmd = [
                sys.executable, '-u', 'train.py',  # -u 参数禁用stdout缓冲
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--lr', str(learning_rate),
                '--num_classes', '10',
                '--optimizer', optimizer,
                '--momentum', str(momentum),
                '--weight_decay', str(weight_decay),
                '--beta1', str(beta1),
                '--beta2', str(beta2),
                '--eps', str(eps)
            ]
            
            self._log_message(training_id, f"执行命令: {' '.join(cmd)}")
            
            # 启动子进程，配置实时输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,  # 无缓冲
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # 确保Python不缓冲输出
            )
            
            # 存储进程引用，用于停止训练
            self.active_trainings[training_id]['process'] = process
            
            current_epoch = 0
            total_epochs = epochs
            
            # 读取进程输出 - 实时处理每一行
            self._log_message(training_id, "开始读取训练输出...")
            
            while True:
                # 检查是否需要停止
                if self.active_trainings[training_id]['status'] == 'stopping':
                    self._log_message(training_id, "用户请求停止训练...")
                    process.terminate()
                    process.wait()
                    self.active_trainings[training_id]['status'] = 'stopped'
                    self._sync_to_database(training_id)
                    return
                
                # 非阻塞读取一行
                line = process.stdout.readline()
                
                # 如果没有输出且进程已结束，退出循环
                if not line and process.poll() is not None:
                    break
                
                # 如果有输出，处理这一行
                if line:
                    line = line.strip()
                    if line:
                        self._log_message(training_id, line)
                        # 解析训练进度和指标
                        self._parse_training_output(training_id, line, total_epochs)
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)
            
            # 等待进程完成
            return_code = process.wait()
            
            if return_code == 0:
                self.active_trainings[training_id]['status'] = 'completed'
                self.active_trainings[training_id]['progress'] = 100.0
                
                # 检查模型文件是否生成
                model_path = 'lenet.pth'
                if os.path.exists(model_path):
                    # 移动模型到静态文件目录
                    model_filename = f"lenet_model_{training_id[:8]}.pth"
                    new_model_path = os.path.join('static', 'models', model_filename)
                    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
                    
                    import shutil
                    shutil.move(model_path, new_model_path)
                    self.active_trainings[training_id]['model_path'] = new_model_path
                    
                self._log_message(training_id, "训练完成！")
                
                # 更新数据库 - 标记为完成
                try:
                    training_record = TrainingRecord.objects.get(training_id=training_id)
                    training_record.completed_at = timezone.now()
                    training_record.save()
                except TrainingRecord.DoesNotExist:
                    pass
                    
            else:
                self.active_trainings[training_id]['status'] = 'error'
                self.active_trainings[training_id]['error_message'] = f"训练进程异常退出，返回码: {return_code}"
                self._log_message(training_id, f"训练失败，返回码: {return_code}")
            
            # 最终同步到数据库
            self._sync_to_database(training_id)
            
            # 从内存中移除已完成的训练（但保留在数据库中）
            if training_id in self.active_trainings:
                # 清理进程引用
                if 'process' in self.active_trainings[training_id]:
                    del self.active_trainings[training_id]['process']
                del self.active_trainings[training_id]
            
        except Exception as e:
            self._log_message(training_id, f"训练出错: {str(e)}")
            if training_id in self.active_trainings:
                self.active_trainings[training_id]['status'] = 'error'
                self.active_trainings[training_id]['error_message'] = str(e)
                self._sync_to_database(training_id)
                
                # 从内存中移除出错的训练
                if 'process' in self.active_trainings[training_id]:
                    del self.active_trainings[training_id]['process']
                del self.active_trainings[training_id]
    
    def _parse_training_output(self, training_id, line, total_epochs):
        """解析训练输出，提取进度和指标"""
        try:
            # 解析Epoch和Step信息: "Epoch [1/1], Step [100/469], Loss: 0.9312"
            epoch_step_match = re.search(r'Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\], Loss: ([\d.]+)', line)
            if epoch_step_match:
                current_epoch = int(epoch_step_match.group(1))
                total_epochs_from_output = int(epoch_step_match.group(2))
                current_step = int(epoch_step_match.group(3))
                total_steps = int(epoch_step_match.group(4))
                current_loss = float(epoch_step_match.group(5))
                
                # 初始化总样本数（只在第一次解析时设置）
                if self.active_trainings[training_id]['total_samples'] == 0:
                    batch_size = self.active_trainings[training_id]['config'].get('batch_size', 64)
                    samples_per_epoch = total_steps * batch_size
                    total_samples = samples_per_epoch * total_epochs_from_output
                    
                    self.active_trainings[training_id]['total_samples'] = total_samples
                    self.active_trainings[training_id]['samples_per_epoch'] = samples_per_epoch
                    
                    self._log_message(training_id, f"总样本数: {total_samples} (每epoch: {samples_per_epoch}, 批次大小: {batch_size})")
                
                # 计算当前已训练样本数
                completed_epochs = current_epoch - 1
                current_epoch_samples = current_step * self.active_trainings[training_id]['config'].get('batch_size', 64)
                current_samples = completed_epochs * self.active_trainings[training_id]['samples_per_epoch'] + current_epoch_samples
                self.active_trainings[training_id]['current_samples'] = current_samples
                
                # 更新基本信息
                self.active_trainings[training_id]['current_epoch'] = current_epoch
                self.active_trainings[training_id]['current_loss'] = current_loss
                
                # 计算精确进度：(已完成的epoch + 当前epoch的step进度) / 总epoch数
                current_epoch_progress = current_step / total_steps
                total_progress = (completed_epochs + current_epoch_progress) / total_epochs_from_output
                self.active_trainings[training_id]['progress'] = total_progress * 100
                
                # 记录每100步的训练损失到step_losses数组
                if current_step % 100 == 0:  # 每100步记录一次
                    step_position = current_step // 100  # 当前epoch中的第几个100步
                    step_label = f"{current_epoch}.{step_position}"  # 如 "1.1", "1.2", "2.1" 等
                    
                    self.active_trainings[training_id]['step_losses'].append(current_loss)
                    self.active_trainings[training_id]['step_labels'].append(step_label)
                
                # 定期同步到数据库（每50步同步一次避免过于频繁）
                if current_step % 50 == 0:
                    self._sync_to_database(training_id)
                
                return  # 已处理完整的step信息，不需要再单独处理loss
            
            
            # 解析epoch训练损失: "Epoch [1/10] Training Loss: 0.1234"
            epoch_train_loss_match = re.search(r'Epoch \[(\d+)/(\d+)\] Training Loss: ([\d.]+)', line)
            if epoch_train_loss_match:
                current_epoch = int(epoch_train_loss_match.group(1))
                train_loss = float(epoch_train_loss_match.group(3))
                
                # 确保历史数组有足够空间
                while len(self.active_trainings[training_id]['epoch_losses']) < current_epoch:
                    self.active_trainings[training_id]['epoch_losses'].append(0.0)
                
                # 记录该epoch的训练损失（epoch_losses[0]是第1个epoch）
                self.active_trainings[training_id]['epoch_losses'][current_epoch - 1] = train_loss
                self._sync_to_database(training_id)
                
                return
            
            # 解析准确率信息: "Test Accuracy of the model is: 92.65 %"
            accuracy_match = re.search(r'Test Accuracy of the model is: ([\d.]+) %', line)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                self.active_trainings[training_id]['current_accuracy'] = accuracy
                
                # 记录准确率到历史数组
                current_epoch = self.active_trainings[training_id].get('current_epoch', 1)  # 默认为1，因为第一次eval是在第1个epoch结束后
                
                # 确保历史数组有足够空间
                while len(self.active_trainings[training_id]['epoch_accuracies']) < current_epoch:
                    self.active_trainings[training_id]['epoch_accuracies'].append(0.0)
                
                # 记录到对应的epoch位置
                if current_epoch > 0:
                    self.active_trainings[training_id]['epoch_accuracies'][current_epoch - 1] = accuracy
                
                self._sync_to_database(training_id)
                return
            
            # 解析验证损失信息: "Test Loss: 0.1234"
            val_loss_match = re.search(r'Test Loss: ([\d.]+)', line)
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
                self.active_trainings[training_id]['current_val_loss'] = val_loss
                
                # 记录验证损失到历史数组
                current_epoch = self.active_trainings[training_id].get('current_epoch', 1)  # 默认为1，因为第一次eval是在第1个epoch结束后
                
                # 确保历史数组有足够空间
                while len(self.active_trainings[training_id]['epoch_val_losses']) < current_epoch:
                    self.active_trainings[training_id]['epoch_val_losses'].append(0.0)
                
                # 记录验证损失到对应epoch位置
                if current_epoch > 0:
                    self.active_trainings[training_id]['epoch_val_losses'][current_epoch - 1] = val_loss
                
                self._sync_to_database(training_id)
                return
            
            # 解析错误样本信息: "Saved error sample 1: True=9, Pred=7"
            error_sample_match = re.search(r'Saved error sample (\d+): True=(\d+), Pred=(\d+)', line)
            if error_sample_match:
                sample_num = int(error_sample_match.group(1))
                true_label = int(error_sample_match.group(2))
                pred_label = int(error_sample_match.group(3))
                
                # 添加错误样本信息
                error_sample_info = {
                    'sample_id': sample_num,
                    'true_label': true_label,
                    'pred_label': pred_label
                }
                self.active_trainings[training_id]['error_samples'].append(error_sample_info)
                
                return
                        
        except Exception as e:
            # 解析失败时记录错误日志，用于调试
            self._log_message(training_id, f"解析输出时出错: {str(e)}, 输出行: {line}")
            pass
    


# 全局训练管理器实例
training_manager = TrainingManager()