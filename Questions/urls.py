from django.contrib import admin
from django.urls import path

from . import views

urlpatterns = [
    path('save-files/', views.fileupload, name='save_staic_fiels'),
    path('Asking/<str:text>/', views.asking, name='Asking')
]
