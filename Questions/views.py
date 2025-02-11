import ast
import collections
import json
import os
import shutil
from itertools import chain
from os import walk
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI

from Main import Parsers

from .forms import FileUploadForm, MakeDirForm, UserValueForm
from .models import History, UserValues


def getalldirs(mypath):
    user = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        user = dirnames
        break
    return user


def allFileformat(mypath, format):
    files = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        csv_files = [file for file in filenames if file.endswith(format)]
        files.extend(csv_files)
        break
    return files


def remove_csv(user):
    user_path = settings.STATIC_UPLOAD_DIR+f'/{user}'

    files = allFileformat(user_path, '.csv')

    for i in files:
        newpath = user_path+f'/{i}'
        if os.path.exists(newpath):
            os.remove(newpath)
            print(f"File '{newpath}' has been deleted.")
        else:
            print(f"File '{newpath}' does not exist.")


def recrate_csvs(user_path, user, parser):
    pdf_files = allFileformat(user_path, '.pdf')
    word_files = allFileformat(user_path, '.docx')
    files = pdf_files + word_files
    for file_name in files:
        file_url = f"{settings.STATIC_UPLOAD_DIR}/{user}/{file_name}"
        fileChunks, fileEmbedings = parser.embedd(file_url)
        parser.SaveCsv(settings.STATIC_UPLOAD_DIR+'/'+user,
                       file_name[:-4], fileEmbedings, fileChunks)


def reembedfiles(user):
    user_path = settings.STATIC_UPLOAD_DIR+f'/{user}'
    user_value = UserValues.objects.filter(user=user).first()
    parser = Parsers(settings.OPENAI_KEY)
    spliter = user_value.splitter
    chunksize = user_value.chunksize
    overlap = user_value.overlap
    parser.SetSpliter(spliter=spliter, chuncksize=chunksize, overlap=overlap)

    remove_csv(user)
    recrate_csvs(user_path, user, parser)


def addfiledata(dic, file_name, chunks, vectors, sim_score):
    dic[file_name] = {
        "chunks": chunks,
        "vectors": vectors,
        "score": sim_score
    }


def sort_data(files_data):
    sorted_files = sorted(files_data.items(),
                          key=lambda item: item[1]['score'], reverse=True)
    top_10_files = sorted_files[:10]
    top_10_chunks = list(chain.from_iterable(
        item[1]['chunks'] for item in top_10_files))

    top_10_vectors = list(chain.from_iterable(
        item[1]['vectors'] for item in top_10_files))

    sorted_files_dict = collections.OrderedDict(top_10_files)

    return (top_10_chunks, top_10_vectors, sorted_files_dict)


def system_file_parser(querry_vector, user):
    mypath = settings.STATIC_UPLOAD_DIR+'/'+user+'/'
    parser = Parsers(settings.OPENAI_KEY)  # ✅

    vectorlist = []
    chunkslist = []
    files = allFileformat(mypath, '.csv')
    files_data = {}
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        # closest_index = parser.cosine_search(vectors, querry_vector)
        top3, similariti_score = parser.cosine_search_top3(
            vectors, querry_vector, 30)
        for j in top3:
            chunkslist.append(chunks[j])
            vectorlist.append(vectors[j])
        if (len(chunkslist) > 0):
            addfiledata(files_data, i, chunkslist,
                        vectorlist, similariti_score)

            # print(files_data[i]['chunks'], files_data[i]['score'])

        chunkslist = []
        vectorlist = []

    top_10_chunks, top_10_vectors, sorted_files_dict = sort_data(files_data)
    # sorted_files = sorted(files_data.items(),
    # key=lambda item: item[1]['score'], reverse=True)
    # top_10_files = sorted_files[:10]
    # top_10_chunks = list(chain.from_iterable(
    # item[1]['chunks'] for item in top_10_files))

    # top_10_vectors = list(chain.from_iterable(
    # item[1]['vectors'] for item in top_10_files))

    # sorted_files_dict = collections.OrderedDict(top_10_files)

    return (top_10_chunks, top_10_vectors, sorted_files_dict)


def addContext(data, message):
    for file_name, content in data.items():
        # print(file_name + '-' * 20)

        chunks = '\n'.join(
            [f"```text\n{chunk}\n```" for chunk in content['chunks']])

        newdic = {
            "role": "system",
            "content": (
                f"The following file: '{
                    file_name}' contains relevant information to support the answer:\n"
                f"{chunks}"
            )
        }
        message.append(newdic)


def addHistory(question_history, answer_history, message):
    question_history = question_history[-10:]
    answer_history = answer_history[-10:]

    # Combine questions and answers into the message
    for index, (q, a) in enumerate(zip(question_history, answer_history)):
        question_entry = {
            "role": "user",
            "content": f"Past Question : {q}"
        }
        answer_entry = {
            "role": "assistant",
            "content": f"Past Answer : {a}"
        }
        message.append(question_entry)
        message.append(answer_entry)


def gettemp(user):
    return UserValues.objects.filter(user=user).first().temp


def context_aware_responses(query, Question_history, Answer_history, data, user):
    # openai.api_key = settings.OPENAI_K
    temp = gettemp(user)
    client = OpenAI(api_key=settings.OPENAI_KEY)
   # print(query, "this is querry", "-"*100)
    # if (len(data) == 0):
    #    return "The context does not contain sufficient information to answer this question.--2"
    messages = [
        {
            "role": "system",
            "content": (
                    "You are an AI Agent designed to assist with answering questions based on the provided context. "
                    "Your behavior and responses are governed by the following rules:\n\n"
                    "1. **Context-Driven Responses Only**:\n"
                    "- You must answer questions **only** based on the given context, user-provided information, or historical interactions.\n"
                    "2. **No External Knowledge or Assumptions**:\n"
                    "- You are not allowed to use knowledge outside the context, assume details, or provide speculative answers.\n\n"
                    "3. **Clear and Concise Answers**:\n"
                    "- Provide clear, accurate, and concise answers based on the available context.\n"
                    "- Avoid verbose explanations unless explicitly requested.\n\n"
                    "4. **Polite and Professional Tone**:\n"
                    "- Maintain a polite and professional tone in all responses.\n\n"
                    "5. **Error Handling**:\n"
                    "- If you encounter ambiguous, contradictory, or invalid input, clarify the issue or state the limitations explicitly.\n\n"
                    "You will now enter a question/answer session. Begin by addressing the user's query based on the context."
            )
        },
        {
            "role": "user",
            "content": f"Current Question: {query}"
        }
    ]

    # {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {  # remove the cuhnks cus it allred add
    #    query}\n\n Past Questino:{Question_history}\n\n past Answer:{Answer_history}"},
    # past quest/ answer make it add context insted of dumping data
    addHistory(Question_history, Answer_history, messages)

    # print('-'*100)
    # print(len(data))
    # print('<>'*50)
    # for i in data:
    # print(data[i]['chunks'])
    # print(len(data[i]['chunks']))
    # print(data[i]['score'])
    # print('-'*100)
    addContext(data, messages)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,  # test on the higer end to the lowest 50-600
        temperature=temp,  # Strict and deterministic responses

    )
    print(messages)
    response_message = response.choices[0].message.content
    return response_message


def unpack_history(history):
    history = list(history)
    if (len(history) > 10):
        history = history[-10:]
    question = []
    answers = []
    for i in history:
        question.append(i[0])
        answers.append(i[1])
    return (question, answers)


def delet_path(user):
    path = os.path.join(settings.STATIC_UPLOAD_DIR, user)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Directory '{path}' deleted successfully.")
    else:
        print(f"Directory '{path}' does not exist.")


def delet_photo(user):
    user_photos_path = os.path.join(
        settings.BASE_DIR, 'static/userphotos', f'{user}.png')
    if os.path.exists(user_photos_path):
        os.remove(user_photos_path)
        print(f"File '{user_photos_path}' has been deleted.")
    else:
        print(f"File '{user_photos_path}' does not exist.")


def delet_user(request, user):
    find = UserValues.objects.filter(user=user)
    find.delete()
    userHistoy = History.objects.filter(sender=user)
    userHistoy.delete()

    users = getalldirs(settings.STATIC_UPLOAD_DIR)

    if (user in users):
        index = users.index(user)
        wanted_users = users[index]
        delet_path(wanted_users)
        delet_photo(user)
    return redirect('/')


def manage_user(request, user):
    if request.method == 'POST':
        form = UserValueForm(request.POST)
        if form.is_valid():  # Validate the form first
            find = UserValues.objects.filter(user=user)
            find.delete()
            chat_message = UserValues(
                user=user,
                splitter=form.cleaned_data['splitter'],
                chunksize=form.cleaned_data['chunksize'],
                overlap=form.cleaned_data['overlap'],
                temp=form.cleaned_data['temp']
            )
            chat_message.save()
            reembedfiles(user)
        return redirect(f'/chat/{user}')
    else:
        form = UserValueForm()

    context = {'user': user, 'form': form}
    return render(request, 'manage-user.html', context)


def manage_users(request):
    mypath = settings.STATIC_UPLOAD_DIR
    upload_dir = os.path.join(settings.BASE_DIR, 'static/userphotos')
    users = getalldirs(mypath)
    userphotos = allFileformat(upload_dir, '.png')
    userphotos.sort()
    users.sort()
    combined = zip(users, userphotos)
    context = {
        'combined': combined
    }

    return render(request, 'manage-users.html', context)


def asking_normal(user, query):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(query)
    chunks, vectors, all_data = system_file_parser(query_vector, user)
    history = user_history(user)
    pastQuestion, pastAnswe = unpack_history(history)
    res = context_aware_responses(
        query, pastQuestion, pastAnswe, all_data, user)
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

    chunk = History.objects.filter(
        sender=user, question=question).first()
    # chunk = []
    # for i in History.objects.filter(sender=user):
    #    chunk.append(i.question)
    #    chunk.append(i.chunks)
    # chunk_list = ast.literal_eval(chunk.chunks)

    # chunk.chunks.replace("'", '"').replace('"s', '\\"s')
    raw_response = chunk.chunks.replace("\'", "\"")

    # Replacing single quotes with double quotes
    # raw_response = chunk.chunks.replace("'", '"')
    # Properly escaping backslashes for double quotes
    # raw_response = raw_response.replace('\\"s', '\\\\"s')

# Now convert the string into a Python list (if needed)
    # json_string = json.dumps(chunk.chunks)
    data = json.loads(raw_response)
    print("this is data")
    files = []
    chunks = []
    for i in data:
        print(i)
        files.append(i['file'])
        chunks.append(i['chunks'][:])

    print("this is data yesysey")
    print(files)
    print("end")

    # data = data*5

    # for i in data:
    #    for j in i['chunks']:
    #        print('<>'*50)
    #        print(j)
    #        print('<>'*50)
    context = {
        #   'chunks': chunk_list,
        'data': data
    }
    return render(request, 'questionchunks.html', context)


def chat(request, user):
    responds = ""
    mypath = settings.STATIC_UPLOAD_DIR+'/'+user
    if (request.method == "POST"):
        text = request.POST.get("question")
        responds, all_data = asking_normal(user, text)
        chat_message = History(
            sender=user, question=text, respons=responds, chunks=unpackdick(all_data))
        chat_message.save()
    pdf_files = allFileformat(mypath, '.pdf')
    word_files = allFileformat(mypath, '.docx')
    files = pdf_files+word_files

    combined = user_history(user)
    context = {
        'user': user,
        'answer': responds,
        'files': files,
        'combined': combined
    }
    return render(request, 'chat.html', context)


"""
okay must make a comment formating patter for the chunks so it can be turend into data that i can parse very simple
so that i dont get a error then must change the chunk.chunks / data= json.loads(raw_response)
"""


def unpackdick(data):

    formatted_data = [
        {"file": filename, "chunks": info["chunks"]}
        for filename, info in data.items()
    ]

    print('_'*100)
    print(formatted_data)
    print('_'*100)

    # return (formatted_data)
    return json.dumps(formatted_data, ensure_ascii=False)


def home(request):
    mypath = settings.STATIC_UPLOAD_DIR
    upload_dir = os.path.join(settings.BASE_DIR, 'static/userphotos')
    users = getalldirs(mypath)
    userphotos = allFileformat(upload_dir, '.png')
    userphotos.sort()
    users.sort()
    combined = zip(users, userphotos)
    context = {
        'combined': combined
    }

    return render(request, 'home.html', context)


def makedir(user_name):
    # print(user_name)
    target_dir = os.path.join(
        settings.BASE_DIR, 'static', 'uploads', user_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Directory {user_name} created at {target_dir}")
    else:
        print(f"Directory {user_name} already exists at {target_dir}")
        os.makedirs(target_dir, exist_ok=True)


def uploadphoto(photoname, photo):
    upload_dir = os.path.join(settings.BASE_DIR, 'static/userphotos')

    # Save the file to the desired path
    fs = FileSystemStorage(location=upload_dir)
    fs.save(photoname, photo)
    answers = allFileformat(upload_dir, '.png')
    print(answers)


@ csrf_exempt
def makedirForm(request):
    if (request.method == "POST"):
        form = MakeDirForm(request.POST, request.FILES)

        if form.is_valid():
            dirname = form.cleaned_data['name']
            photo = form.cleaned_data['photo']  # The uploaded image file

            makedir(dirname)
            print(dirname)

            photoname = dirname+photo.name[-4:]
            uploadphoto(photoname, photo)
            return redirect(f"/Manage-User/{dirname}")
    form = MakeDirForm()
    return render(request, 'upload_file.html', {'form': form})


def save_file(uploaded_file, user):
    fs = FileSystemStorage(location=settings.STATIC_UPLOAD_DIR+f'/{user}')
    fs.save(uploaded_file.name, uploaded_file)

    file_url = f"{settings.STATIC_UPLOAD_DIR}/{user}/{uploaded_file.name}"
    parser = Parsers(settings.OPENAI_KEY)
    user_value = UserValues.objects.filter(user=user).first()
    spliter = user_value.splitter
    chunksize = user_value.chunksize
    overlap = user_value.overlap
    # print(f'info about user {user} chunksize {
    #      chunksize} overlap {overlap} spliter {spliter}')
    parser.SetSpliter(spliter=spliter, chuncksize=chunksize, overlap=overlap)

    fileChunks, fileEmbedings = parser.embedd(file_url)

    parser.SaveCsv(settings.STATIC_UPLOAD_DIR+'/'+user,
                   uploaded_file.name, fileEmbedings, fileChunks)
    return fileEmbedings


@ csrf_exempt
def fileupload(request, user):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        my_file = Path(
            f"{settings.STATIC_UPLOAD_DIR}/{user}/{uploaded_file.name}")
        if not my_file.is_file():
            save_file(uploaded_file, user)
            return redirect(f"/chat/{user}/")
        else:
            return JsonResponse({'success': False, 'data': "The File Existers allready"})
    else:
        form = FileUploadForm()
    return render(request, 'save-static.html', {'form': form})
