from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/<str:user>/', views.chat, name='chat'),

    path('makedir/', views.makedirForm, name='makedir'),


    path('save-files/<str:user>', views.fileupload, name='save-files'),

    path('Manage-Users/', views.manage_users, name='Mange-Users'),
    path('Manage-User/<str:user>', views.manage_user, name='Mange-user'),

    path('delete-user/<str:user>', views.delet_user, name='delet-user'),

    path('get-chunks/<str:user>/<str:question>',
         views.getchunksforQuestin, name='chunks')
]
