import os
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .forms import FileUploadForm, MakeDirForm, UserValueForm
from .models import Chunk, History, UserValues
from .services.AI_service import asking_normal, context_aware_responses, get_save_output
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
from .services.user_services import makeuser, sortedUsers, user_history



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
            makeuser(user, form)
        return redirect(f"/chat/{user}")
    else:
        form = UserValueForm()

    context = {"user": user, "form": form}
    return render(request, "manage-user.html", context)


def manage_users(request):
    combined = sortedUsers()
    context = {"combined": combined}
    return render(request, "manage-users.html", context)


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


def chat(request, user):
    responds = ""
    mypath = settings.STATIC_UPLOAD_DIR + "/" + user
    if request.method == "POST":
        text = request.POST.get("question")

        responds, all_data = get_save_output(user, text)
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
