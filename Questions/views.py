import os
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .forms import FileUploadForm, MakeDirForm, UserValueForm
from .models import Chunk, History, UserValues
from .services.AI_service import context_aware_responses
from .services.fileSystem_service import (
    addfiledata,
    allFileformat,
    delet_path,
    delet_photo,
    getalldirs,
    makedir,
    reembedfiles,
    save_file,
    sort_data,
    system_file_parser,
)
from .services.parser_service import Parsers


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


def delet_user(request, user):
    find = UserValues.objects.filter(user=user)
    find.delete()
    userHistoy = History.objects.filter(sender=user)
    userHistoy.delete()

    users = getalldirs(settings.STATIC_UPLOAD_DIR)

    if user in users:
        index = users.index(user)
        wanted_users = users[index]
        delet_path(wanted_users)
        delet_photo(user)
    return redirect("/")


def manage_user(request, user):
    if request.method == "POST":
        form = UserValueForm(request.POST)
        if form.is_valid():  # Validate the form first
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
        return redirect(f"/chat/{user}")
    else:
        form = UserValueForm()

    context = {"user": user, "form": form}
    return render(request, "manage-user.html", context)


def manage_users(request):
    mypath = settings.STATIC_UPLOAD_DIR
    upload_dir = os.path.join(settings.BASE_DIR, "static/userphotos")
    users = getalldirs(mypath)
    userphotos = allFileformat(upload_dir, ".png")
    userphotos.sort()
    users.sort()
    combined = zip(users, userphotos)
    context = {"combined": combined}

    return render(request, "manage-users.html", context)


def asking_normal(user, query):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(query)
    chunks, vectors, all_data = system_file_parser(query_vector, user)
    history = user_history(user)
    pastQuestion, pastAnswe = unpack_history(history)
    res = context_aware_responses(query, pastQuestion, pastAnswe, all_data, user)
    return (res, all_data)


def user_history(user):
    chat_history = History.objects.filter(sender=user)
    question = []
    answers = []
    for i in chat_history:
        question.append(i.question)
        answers.append(i.respons)

    return zip(question, answers)


def getchunksforQuestin(request, user, question):

    instance = History.objects.filter(sender=user, question=question).first()

    data = []
    for i in instance.chunks.all():
        data.append(i.chunk_text)

    print("this is data and this is the len ", len(data))
    print(data)
    context = {
        #   'chunks': chunk_list,
        "data": data
    }
    return render(request, "test.html", context)


def saveHitoryChunsk(instance, data):

    print("$" * 50)
    for i in unpackdick(data):
        for j in i:
            print(j[:50])
            Chunk.objects.create(history=instance, chunk_text=j)
    print("=" * 50)
    for i in instance.chunks.all():
        print(i.chunk_text[:50])

    print("$" * 50)
    return None


def chat(request, user):
    responds = ""
    mypath = settings.STATIC_UPLOAD_DIR + "/" + user
    if request.method == "POST":
        text = request.POST.get("question")
        responds, all_data = asking_normal(user, text)
        chat_message = History(
            # , chunks=unpackdick(all_data))
            sender=user,
            question=text,
            respons=responds,
        )
        chat_message.save()
        saveHitoryChunsk(chat_message, all_data)
    pdf_files = allFileformat(mypath, ".pdf")
    word_files = allFileformat(mypath, ".docx")
    files = pdf_files + word_files

    combined = user_history(user)
    context = {"user": user, "answer": responds, "files": files, "combined": combined}
    return render(request, "chat.html", context)


"""
okay must make a comment formating patter for the chunks so it can be turend into data that i can parse very simple
so that i dont get a error then must change the chunk.chunks / data= json.loads(raw_response)
"""


def unpackdick(data):

    #    formatted_data = [
    #    {"file": filename, "chunks": info["chunks"]}

    #    for filename, info in data.items()
    # ]
    formatted_data = [info["chunks"] for filename, info in data.items()]

    return formatted_data


def home(request):
    mypath = settings.STATIC_UPLOAD_DIR
    upload_dir = os.path.join(settings.BASE_DIR, "static/userphotos")
    users = getalldirs(mypath)
    userphotos = allFileformat(upload_dir, ".png")
    userphotos.sort()
    users.sort()
    combined = zip(users, userphotos)
    context = {"combined": combined}

    return render(request, "home.html", context)


def uploadphoto(photoname, photo):
    upload_dir = os.path.join(settings.BASE_DIR, "static/userphotos")

    fs = FileSystemStorage(location=upload_dir)
    fs.save(photoname, photo)
    answers = allFileformat(upload_dir, ".png")
    print(answers)


@csrf_exempt
def makedirForm(request):
    if request.method == "POST":
        form = MakeDirForm(request.POST, request.FILES)

        if form.is_valid():
            dirname = form.cleaned_data["name"]
            photo = form.cleaned_data["photo"]  # The uploaded image file

            makedir(dirname)
            print(dirname)

            photoname = dirname + photo.name[-4:]
            uploadphoto(photoname, photo)
            return redirect(f"/Manage-User/{dirname}")
    form = MakeDirForm()
    return render(request, "upload_file.html", {"form": form})



@csrf_exempt
def fileupload(request, user):
    if request.method == "POST" and request.FILES["file"]:
        uploaded_file = request.FILES["file"]
        my_file = Path(f"{settings.STATIC_UPLOAD_DIR}/{user}/{uploaded_file.name}")
        if not my_file.is_file():
            save_file(uploaded_file, user)
            return redirect(f"/chat/{user}/")
        else:
            return JsonResponse({"success": False, "data": "The File Existers allready"})
    else:
        form = FileUploadForm()
    return render(request, "save-static.html", {"form": form})
