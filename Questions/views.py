import os
from os import walk
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)
from sklearn.metrics.pairwise import cosine_similarity

from Main import Parsers

from .forms import FileUploadForm
from .models import TestUPFILE

"""
Need to refactor the code and remvoe fileEmbedings lot of shit just rember to do this this week sundaytusday

              """

# Create your views here.


def index(reuqest):
    return HttpResponse("Hello, world. You're at the polls index.")


def hello(re):
    my_list = ['hello', 'this is somthing quite intresitn',
               'this on the other hand is quite more simple ', 59000]
    return render(request=re, template_name="home.html", context={'my_list': my_list})


@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Redirect to fileEmbedings success page or display fileEmbedings success message
            return redirect('file_upload_success')
            # return redirect('save_staic_fiels')
    else:
        form = FileUploadForm()
    return render(request, 'upload_file.html', {'form': form})


def goThroEveryfile(querry_vector):
    mypath = settings.STATIC_UPLOAD_DIR+'/'
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150, length_function=len
    )
    fileEmbedings = Parsers(settings.OPENAI_KEY,
                            text_splitter)

    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    allvalus = []
    allchunks = []
    for i in f:
        if i[-4:] == '.csv':
            a = fileEmbedings.ReadFromFile(mypath+i)
            id = fileEmbedings.cosine_search(a[1], querry_vector)
            allvalus.append(a[1][id])
            allchunks.append(a[0][id])
    return (allchunks, allvalus)


def asking(request, text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150, length_function=len
    )
    fileEmbedings = Parsers(settings.OPENAI_KEY,text_splitter)
    query_vector = fileEmbedings.embedquerry(text)

    k = goThroEveryfile(query_vector)

    b = fileEmbedings.cosine_search(k[1], query_vector)
    return HttpResponse(f"what ever you say bro -> {k[0][b]}")


@ csrf_exempt
def save_files(request):
    if request.method == 'POST' and request.FILES['file']:

        uploaded_file = request.FILES['file']

        # Define the target directory in the static folder
        # target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads')
        # os.makedirs(target_dir, exist_ok=True)
        my_file = Path(f"{settings.STATIC_UPLOAD_DIR}/{uploaded_file.name}")
        if not my_file.is_file():
            fs = FileSystemStorage(location=settings.STATIC_UPLOAD_DIR)
            fs.save(uploaded_file.name, uploaded_file)
            file_url = f"{settings.STATIC_UPLOAD_DIR}/{uploaded_file.name}"
            spliter = RecursiveCharacterTextSplitter(
                chunk_size=600, chunk_overlap=150, length_function=len
            )
            tParser = Parsers(settings.OPENAI_KEY, spliter)
            fileEmbedings = tParser.embedd(file_url)
            tParser.SaveCsv(settings.STATIC_UPLOAD_DIR,
                            uploaded_file.name[:-4], fileEmbedings, tParser.getChunks())
            return JsonResponse({'success': True, 'data': fileEmbedings})
        else:
            return JsonResponse({'success': False, 'data': "this file allredy existe mf "})
    else:
        form = FileUploadForm()
    return render(request, 'save-static.html', {'form': form})


def upload_success(request):
    return render(request, 'upload_success.html')


def show(request):
    l1 = TestUPFILE.objects.all()
    text = '||'
    for i in l1:
        text += i.name
        text += '||'
    text += str(len(l1))

    return HttpResponse(text)
