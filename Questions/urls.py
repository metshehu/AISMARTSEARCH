from django.contrib import admin
from django.urls import path

from . import views
from .views import upload_file

urlpatterns = [
    path("", views.index, name="index"),
    path("12", views.hello, name="hello"),
    path('upload/', upload_file, name='upload_file'),
    path('upload-success/', views.upload_success, name='file_upload_success'),
    path('showfiles', views.show, name='show'),
    path('save-files/', views.save_files, name='save_staic_fiels'),
    path('Asking/<str:text>/', views.asking, name='Asking')




]
