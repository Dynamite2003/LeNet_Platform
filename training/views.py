from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .train_manager import training_manager

def home(request):
    """主页视图"""
    return render(request, 'training/home.html')

def train(request):
    """训练页面视图"""
    if request.method == 'GET':
        return render(request, 'training/train.html')

@csrf_exempt
@require_http_methods(["POST"])
def start_training(request):
    """开始训练API"""
    try:
        data = json.loads(request.body)
        
        # 验证训练参数
        config = {
            'epochs': int(data.get('epochs', 10)),
            'batch_size': int(data.get('batch_size', 64)),
            'learning_rate': float(data.get('learning_rate', 0.001)),
            'optimizer': data.get('optimizer', 'sgd')
        }
        
        # 添加优化器参数
        if config['optimizer'] == 'adam':
            config.update({
                'beta1': float(data.get('beta1', 0.9)),
                'beta2': float(data.get('beta2', 0.999)),
                'eps': float(data.get('eps', 1e-8)),
                'weight_decay': float(data.get('weight_decay', 1e-4))
            })
        else:  # SGD
            config.update({
                'momentum': float(data.get('momentum', 0.9)),
                'weight_decay': float(data.get('weight_decay', 1e-4))
            })
        
        # 参数验证
        if not (1 <= config['epochs'] <= 100):
            return JsonResponse({'status': 'error', 'message': '训练轮数必须在1-100之间'})
        
        if config['batch_size'] not in [16, 32, 64, 128, 256]:
            return JsonResponse({'status': 'error', 'message': '批次大小必须是16, 32, 64, 128, 256中的一个'})
        
        if not (0.0 < config['learning_rate'] <= 1.0):
            return JsonResponse({'status': 'error', 'message': '学习率必须在0-1之间（不包括0）'})
        
        if config['optimizer'] not in ['sgd', 'adam']:
            return JsonResponse({'status': 'error', 'message': '优化器必须是SGD或Adam'})
        
        # 优化器参数验证
        if config['optimizer'] == 'adam':
            if not (0.0 <= config['beta1'] < 1.0):
                return JsonResponse({'status': 'error', 'message': 'Beta1必须在0-1之间（不包括1）'})
            if not (0.0 <= config['beta2'] < 1.0):
                return JsonResponse({'status': 'error', 'message': 'Beta2必须在0-1之间（不包括1）'})
            if not (0.0 < config['eps'] <= 0.01):
                return JsonResponse({'status': 'error', 'message': 'Epsilon必须在0-0.01之间（不包括0）'})
        else:  # SGD
            if not (0.0 <= config['momentum'] <= 1.0):
                return JsonResponse({'status': 'error', 'message': 'Momentum必须在0-1之间'})
        
        if not (0.0 <= config['weight_decay'] <= 0.1):
            return JsonResponse({'status': 'error', 'message': 'Weight Decay必须在0-0.1之间'})
        
        # 开始训练
        training_id = training_manager.start_training(config)
        
        return JsonResponse({
            'status': 'success',
            'message': '训练已开始',
            'training_id': training_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '无效的JSON数据'})
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': f'参数错误: {str(e)}'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'服务器错误: {str(e)}'})

@require_http_methods(["GET"])
def training_status(request, training_id):
    """获取训练状态API"""
    status = training_manager.get_training_status(training_id)
    
    if status is None:
        return JsonResponse({'status': 'error', 'message': '训练任务不存在'})
    
    # 创建一个可序列化的副本，移除process对象
    clean_status = status.copy()
    if 'process' in clean_status:
        del clean_status['process']
    
    return JsonResponse({
        'status': 'success',
        'data': clean_status
    })

@csrf_exempt
@require_http_methods(["POST"])
def stop_training(request, training_id):
    """停止训练API"""
    success = training_manager.stop_training(training_id)
    
    if success:
        return JsonResponse({'status': 'success', 'message': '训练停止请求已发送'})
    else:
        return JsonResponse({'status': 'error', 'message': '无法停止训练，可能训练已完成或不存在'})

@require_http_methods(["GET"])
def training_logs(request, training_id):
    """获取训练日志API"""
    status = training_manager.get_training_status(training_id)
    
    if status is None:
        return JsonResponse({'status': 'error', 'message': '训练任务不存在'})
    
    # 获取最新的日志（可以分页）
    logs = status.get('logs', [])
    offset = int(request.GET.get('offset', 0))
    limit = int(request.GET.get('limit', 50))
    
    return JsonResponse({
        'status': 'success',
        'logs': logs[offset:offset + limit],
        'total': len(logs)
    })

@require_http_methods(["GET"])
def all_trainings(request):
    """获取所有训练任务API"""
    trainings = training_manager.get_all_trainings()
    return JsonResponse({
        'status': 'success',
        'data': trainings
    })

@require_http_methods(["GET"])
def training_history(request):
    """获取训练历史记录API"""
    all_records = training_manager.get_all_training_records()
    return JsonResponse({
        'status': 'success',
        'data': all_records
    })

def results(request):
    """结果展示页面视图"""
    # 获取用户选择的训练记录ID，如果没有则默认选择最新的
    selected_training_id = request.GET.get('training_id', None)
    
    # 获取所有训练记录
    all_training_records = training_manager.get_all_training_records()
    
    # 选择要显示的训练记录
    selected_training = None
    if selected_training_id:
        # 用户指定了特定的训练记录
        for training in all_training_records:
            if training['id'] == selected_training_id:
                selected_training = training
                break
    
    # 如果没有找到指定的记录，或者用户没有指定，则选择最新完成的训练
    if not selected_training:
        completed_trainings = [t for t in all_training_records if t['status'] == 'completed']
        if completed_trainings:
            # 按开始时间排序，选择最新的
            selected_training = max(completed_trainings, key=lambda x: x.get('start_time', ''))
    
    context = {
        'selected_training': json.dumps(selected_training) if selected_training else None,
        'selected_training_obj': selected_training,
        'has_training_data': selected_training is not None,
        'all_training_records': all_training_records,
        'selected_training_id': selected_training_id or (selected_training['id'] if selected_training else None)
    }
    
    return render(request, 'training/results.html', context)
