from django.db import models
from django.utils import timezone
import json

class TrainingRecord(models.Model):
    """训练记录模型"""
    
    # 训练基本信息
    training_id = models.CharField(max_length=36, unique=True, help_text="训练任务的唯一ID")
    status = models.CharField(max_length=20, default='preparing', 
                             choices=[
                                 ('preparing', '准备中'),
                                 ('running', '运行中'),
                                 ('completed', '已完成'),
                                 ('error', '出错'),
                                 ('stopped', '已停止'),
                                 ('stopping', '停止中'),
                             ])
    
    # 时间信息
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # 训练配置（JSON字段）
    config = models.JSONField(help_text="训练配置参数")
    
    # 训练进度
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=10)
    progress = models.FloatField(default=0.0, help_text="训练进度百分比")
    
    # 当前指标
    current_loss = models.FloatField(default=0.0)
    current_accuracy = models.FloatField(default=0.0)
    current_val_loss = models.FloatField(default=0.0)
    
    # 历史数据（JSON字段）
    epoch_losses = models.JSONField(default=list, help_text="每个epoch的训练损失")
    epoch_accuracies = models.JSONField(default=list, help_text="每个epoch的验证准确率")
    epoch_val_losses = models.JSONField(default=list, help_text="每个epoch的验证损失")
    step_losses = models.JSONField(default=list, help_text="每100步的训练损失")
    step_labels = models.JSONField(default=list, help_text="每100步的标签")
    
    # 样本计数
    total_samples = models.IntegerField(default=0)
    current_samples = models.IntegerField(default=0)
    samples_per_epoch = models.IntegerField(default=0)
    
    # 错误样本信息
    error_samples = models.JSONField(default=list, help_text="错误样本信息")
    
    # 其他信息
    error_message = models.TextField(blank=True, null=True)
    model_path = models.CharField(max_length=255, blank=True, null=True)
    logs = models.JSONField(default=list, help_text="训练日志")
    
    class Meta:
        ordering = ['-created_at']  # 按创建时间倒序排列
        
    def __str__(self):
        return f"训练记录 {self.training_id[:8]} - {self.status}"
    
    def get_config_display(self):
        """获取配置的显示字符串"""
        config = self.config
        return f"Epochs: {config.get('epochs', 'N/A')}, LR: {config.get('learning_rate', 'N/A')}, Optimizer: {config.get('optimizer', 'N/A').upper()}"
    
    def get_duration(self):
        """获取训练持续时间"""
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}小时{minutes}分钟{seconds}秒"
            elif minutes > 0:
                return f"{minutes}分钟{seconds}秒"
            else:
                return f"{seconds}秒"
        return "N/A"
    
    def to_dict(self):
        """转换为字典格式，用于JSON序列化"""
        return {
            'id': self.training_id,
            'status': self.status,
            'config': self.config,
            'start_time': self.started_at.isoformat() if self.started_at else self.created_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': self.current_loss,
            'current_accuracy': self.current_accuracy,
            'current_val_loss': self.current_val_loss,
            'epoch_losses': self.epoch_losses,
            'epoch_accuracies': self.epoch_accuracies,
            'epoch_val_losses': self.epoch_val_losses,
            'step_losses': self.step_losses,
            'step_labels': self.step_labels,
            'logs': self.logs,
            'progress': self.progress,
            'error_message': self.error_message,
            'model_path': self.model_path,
            'total_samples': self.total_samples,
            'current_samples': self.current_samples,
            'samples_per_epoch': self.samples_per_epoch,
            'error_samples': self.error_samples,
            'duration': self.get_duration(),
        }
