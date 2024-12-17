import os
from os import walk
from pathlib import Path

import numpy as np
import openai
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
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from Main import Parsers

from .forms import FileUploadForm

"""
Need to refactor the code and remvoe fileEmbedings lot of shit just rember to do this this week sundaytusday

"""


def allCsvfiles(mypath):
    files = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        csv_files = [file for file in filenames if file.endswith('.csv')]
        files.extend(csv_files)
        break
    return files


def system_file_parser(querry_vector):
    mypath = settings.STATIC_UPLOAD_DIR+'/'
    parser = Parsers(settings.OPENAI_KEY)
    vectorlist = []
    chunkslist = []
    files = allCsvfiles(mypath)
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        closest_index = parser.cosine_search(vectors, querry_vector)
        vectorlist.append(vectors[closest_index])
        chunkslist.append(chunks[closest_index])
    return (chunkslist, vectorlist)


def openaitest(chunks, query):
    # openai.api_key = settings.OPENAI_K
    client = OpenAI(api_key=settings.OPENAI_KEY)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150  # test on the higer end to the lowest 50-600
    )

    response_message = response.choices[0].message.content
    return response_message
    # return response['choices'][0]['message']['content'].strip()


def asking(request, text):

    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)

    k = system_file_parser(query_vector)

    # b = fileEmbedings.cosine_search(k[1], query_vector)
    res = openaitest(k[0], text)
    return HttpResponse(f"what ever you say bro -> {res}")


def save_file(uploaded_file):

    # target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads')
    # os.makedirs(target_dir, exist_ok=True)
    fs = FileSystemStorage(location=settings.STATIC_UPLOAD_DIR)
    fs.save(uploaded_file.name, uploaded_file)

    file_url = f"{settings.STATIC_UPLOAD_DIR}/{uploaded_file.name}"
    parser = Parsers(settings.OPENAI_KEY)
    fileChunks, fileEmbedings = parser.embedd(file_url)
    parser.SaveCsv(settings.STATIC_UPLOAD_DIR,
                   uploaded_file.name[:-4], fileEmbedings, fileChunks)
    return fileEmbedings


@ csrf_exempt
def fileupload(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        my_file = Path(f"{settings.STATIC_UPLOAD_DIR}/{uploaded_file.name}")
        if not my_file.is_file():
            vectors = save_file(uploaded_file)
            return JsonResponse({'success': True, 'data': vectors})
        else:
            return JsonResponse({'success': False, 'data': "The File Existers allready"})
    else:
        form = FileUploadForm()
    return render(request, 'save-static.html', {'form': form})
