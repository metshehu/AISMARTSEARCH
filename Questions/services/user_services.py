import os

from django.conf import settings
from django.shortcuts import redirect

from Questions.models import Chunk, History, UserValues

from .fileSystem_service import allFileformat, delet_path, delet_photo, getalldirs, reembedfiles


def makeuser(user, form):
    find = UserValues.objects.filter(user=user)
    find.delete()
    chat_message = UserValues(
        user=user,
        splitter=form.cleaned_data["splitter"],
        chunksize=form.cleaned_data["chunksize"],
        overlap=form.cleaned_data["overlap"],
        temp=form.cleaned_data["temp"],
    )
    chat_message.save()
    reembedfiles(user)


def sortedUsers():
    mypath = settings.STATIC_UPLOAD_DIR
    upload_dir = os.path.join(settings.BASE_DIR, "static/userphotos")
    users = getalldirs(mypath)
    userphotos = allFileformat(upload_dir, ".png")
    userphotos.sort()
    users.sort()
    return zip(users, userphotos)


def user_history(user):
    chat_history = History.objects.filter(sender=user)
    question = []
    answers = []
    for i in chat_history:
        question.append(i.question)
        answers.append(i.respons)

    return zip(question, answers)


def unpackdick(data):
    formatted_data = [info["chunks"] for filename, info in data.items()]
    return formatted_data


def unpack_history(history):
    history = list(history)
    if len(history) > 10:
        history = history[-10:]
    question = []
    answers = []
    for i in history:
        question.append(i[0])
        answers.append(i[1])
    return (question, answers)



def saveHitoryChunsk(instance, data):
    for i in unpackdick(data):
        for j in i:
            Chunk.objects.create(history=instance, chunk_text=j)

    return None
