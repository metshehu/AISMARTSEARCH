from django.contrib import admin
from django.urls import path

from . import views

urlpatterns = [
    path('save-files/<str:mydir>', views.fileupload, name='save-files'),
    path('Asking/<str:mydir>/<str:text>', views.asking, name='Asking'),
    path('frontend/<str:mydir>', views.front, name='frontend'),
    path('chat/<str:dir>/', views.chat, name='chat'),
    path('home/', views.home, name='home'),
    path('dirs/<str:dir_name>/', views.makedir, name='dirs')
]
