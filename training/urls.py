from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('train/', views.train, name='train'),
    path('results/', views.results, name='results'),
    
    # API endpoints
    path('api/start-training/', views.start_training, name='start_training'),
    path('api/training-status/<str:training_id>/', views.training_status, name='training_status'),
    path('api/stop-training/<str:training_id>/', views.stop_training, name='stop_training'),
    path('api/training-logs/<str:training_id>/', views.training_logs, name='training_logs'),
    path('api/all-trainings/', views.all_trainings, name='all_trainings'),
    path('api/training-history/', views.training_history, name='training_history'),
]