import collections
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

"""
Need to refactor the code and remvoe fileEmbedings lot of shit just rember to do this this week sundaytusday

"""
""""
endpoint = os.getenv("OPENAI_API_KEY")
print(endpoint)
llm = ChatOpenAI(
    api_key=endpoint
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "your answers Amarican RedNeck"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

respones = chain.invoke(
    {"input": "Can you roast me based on everything you know about me"})
somthing to think about later
"""


def getalldirs(mypath):
    dir = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        dir = dirnames
        break
    return dir


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


def reembedfiles(user):

    user_path = settings.STATIC_UPLOAD_DIR+f'/{user}'
    user_value = UserValues.objects.filter(user=user).first()
    parser = Parsers(settings.OPENAI_KEY)
    spliter = user_value.splitter
    chunksize = user_value.chunksize
    overlap = user_value.overlap
    parser.SetSpliter(spliter=spliter, chuncksize=chunksize, overlap=overlap)

    remove_csv(user)

    files = allFileformat(user_path, '.pdf')
    for file_name in files:
        file_url = f"{settings.STATIC_UPLOAD_DIR}/{user}/{file_name}"
        fileChunks, fileEmbedings = parser.embedd(file_url)
        parser.SaveCsv(settings.STATIC_UPLOAD_DIR+'/'+user,
                       file_name[:-4], fileEmbedings, fileChunks)


def ogsystem_file_parser(querry_vector, mydir):
    mypath = settings.STATIC_UPLOAD_DIR+'/'+mydir+'/'
    parser = Parsers(settings.OPENAI_KEY)

    vectorlist = []
    chunkslist = []
    files = allFileformat(mypath, '.csv')
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        closest_index = parser.cosine_search(vectors, querry_vector)
        vectorlist.append(vectors[closest_index])
        chunkslist.append(chunks[closest_index])
    return (chunkslist, vectorlist)


def addfiledata(dic, file_name, chunks, vectors, sim_score):
    dic[file_name] = {
        "chunks": chunks,
        "vectors": vectors,
        "score": sim_score
    }


def system_file_parser(querry_vector, mydir):
    mypath = settings.STATIC_UPLOAD_DIR+'/'+mydir+'/'
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

    sorted_files = sorted(files_data.items(),
                          key=lambda item: item[1]['score'], reverse=True)
    top_10_files = sorted_files[:10]
    top_10_chunks = list(chain.from_iterable(
        item[1]['chunks'] for item in top_10_files))

    top_10_vectors = list(chain.from_iterable(
        item[1]['vectors'] for item in top_10_files))

    sorted_files_dict = collections.OrderedDict(top_10_files)

    return (top_10_chunks, top_10_vectors, sorted_files_dict)


def addContextother(data, messages):
    """
    Adds context to the messages by iterating through the given data.
    """
    for filename, file_data in data.items():
        # Concatenate chunks from the file
        text = "\n".join(file_data.get('chunks', []))

        # Add a strict instruction regarding the use of this context
        new_entry = {
            "role": "system",
            "content": (
                f"The following file: '{
                    filename}' contains the relevant information: \n"
                f"```text\n{text}\n```\n"
                "You must strictly rely on this information to answer questions. "
                "If this context is insufficient to answer the question, respond with: "
                "'The context does not contain sufficient information to answer this question.'"
            )
        }
        messages.append(new_entry)


def addContext(data, message):
    for i in data:
        print(i+'-'*20)
        text = ''
        for j in data[i]['chunks']:
            text += f"```text {j}```"

        newdic = {"role": "system", "content": f"The following file: '{
            i}'contains the following relevant information to support the answer:{text}"}

        # {"role": "system", "content": "The following file": "Computer science and engineering sylabus” contains the following relevant information to support the answer:"}
        message.append(newdic)


def addHistory(question_history, answer_history, message):
    # Limit to the last 10 interactions for brevity
    question_history = question_history[-10:]
    answer_history = answer_history[-10:]

    # Combine questions and answers into the message
    for index, (q, a) in enumerate(zip(question_history, answer_history)):
        question_entry = {
            "role": "user",
            "content": f"Past Question {index + 1}: {q}"
        }
        answer_entry = {
            "role": "assistant",
            "content": f"Past Answer {index + 1}: {a}"
        }
        message.append(question_entry)
        message.append(answer_entry)


# def addHistroy(question_history, answer_histry, message):
#    index = 0
#    question_history = question_history[-10:]
#    answer_histry = answer_histry[-10:]

# for q, a in zip(question_history, answer_histry):
#        question_dic = {
#            "role": "user", "content": f" this is a past question number {index} {q}"}
#        answer_dic = {"role": "assistant",
#                      "content": f" this is a past answer for question number {index} {a}"}

#        message.append(question_dic)
#        message.append(answer_dic)
#        index += 1

def gettemp(user):
    return UserValues.objects.filter(user=user).first().temp


def context_aware_responses(query, Question_history, Answer_history, data, user):
    # openai.api_key = settings.OPENAI_K
    temp = gettemp(user)
    client = OpenAI(api_key=settings.OPENAI_KEY)
    print(query, "this is querry", "-"*100)
    if (len(data) == 0):
        return "The context does not contain sufficient information to answer this question.--2"

    messages = [
        {"role": "system", "content": (
            "You are a context-aware search engine. You must return responses **only** based on the provided context  and the user's past interactions. "
            #            "You are not allowed to infer or assume answers if the context does not provide sufficient information. "
            #            "If the context does not contain sufficient information to answer the question, respond only with: "
            #            "'The context does not contain sufficient information to answer this question.' "
            #            "Do not provide additional information, guesses, or speculative answers."
        )},
        {"role": "user", "content": f"current Question: {query}"},
    ]

    # {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {  # remove the cuhnks cus it allred add
    #    query}\n\n Past Questino:{Question_history}\n\n past Answer:{Answer_history}"},
    # past quest/ answer make it add context insted of dumping data
    addHistory(Question_history, Answer_history, messages)

    print('-'*100)

    print(len(data))
    print('<>'*50)
    for i in data:
        print(data[i]['chunks'])
        print(len(data[i]['chunks']))
        print(data[i]['score'])
    print('-'*100)
    addContext(data, messages)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,  # test on the higer end to the lowest 50-600
        temperature=temp,  # Strict and deterministic responses

    )
    response_message = response.choices[0].message.content
    return response_message
    # return response['choices'][0]['message']['content'].strip()


def asking(request, mydir, text):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)
    chunks, vectors = system_file_parser(query_vector, mydir)
    history = user_history(mydir)
    res = context_aware_responses(chunks, text)
    return HttpResponse(f"{res}")


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


def asking_normal(mydir, query):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(query)
    chunks, vectors, all_data = system_file_parser(query_vector, mydir)
    history = user_history(mydir)
    pastQuestion, pastAnswe = unpack_history(history)
    res = context_aware_responses(
        query, pastQuestion, pastAnswe, all_data, mydir)
    return res


def user_history(user):
    chat_history = History.objects.filter(sender=user)
    question = []
    answers = []
    for i in chat_history:
        question.append(i.question)
        answers.append(i.respons)

    return zip(question, answers)


def chat(request, dir):
    responds = ""
    mypath = settings.STATIC_UPLOAD_DIR+'/'+dir
    if (request.method == "POST"):
        text = request.POST.get("question")
        responds = asking_normal(dir, text)
        chat_message = History(
            sender=dir, question=text, respons=responds)
        chat_message.save()
    files = allFileformat(mypath, '.pdf')
    combined = user_history(dir)
    context = {
        'dir': dir,
        'answer': responds,
        'files': files,
        'combined': combined
    }
    return render(request, 'chat.html', context)


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


def makedir(dir_name):
    print(dir_name)
    target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', dir_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Directory {dir_name} created at {target_dir}")
    else:
        print(f"Directory {dir_name} already exists at {target_dir}")
        os.makedirs(target_dir, exist_ok=True)


def uploadphoto(photoname, photo):
    # Define the full file path
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
    # return render(request,'home.html')


def save_file(uploaded_file, dir):
    # target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads')
    # os.makedirs(target_dir, exist_ok=True)
    fs = FileSystemStorage(location=settings.STATIC_UPLOAD_DIR+f'/{dir}')
    fs.save(uploaded_file.name, uploaded_file)

    file_url = f"{settings.STATIC_UPLOAD_DIR}/{dir}/{uploaded_file.name}"
    parser = Parsers(settings.OPENAI_KEY)
    user_value = UserValues.objects.filter(user=dir).first()
    spliter = user_value.splitter
    chunksize = user_value.chunksize
    overlap = user_value.overlap
    print(f'info about user {dir} chunksize {
          chunksize} overlap {overlap} spliter {spliter}')
    parser.SetSpliter(spliter=spliter, chuncksize=chunksize, overlap=overlap)

    fileChunks, fileEmbedings = parser.embedd(file_url)
    parser.SaveCsv(settings.STATIC_UPLOAD_DIR+'/'+dir,
                   uploaded_file.name[:-4], fileEmbedings, fileChunks)
    return fileEmbedings


@ csrf_exempt
def fileupload(request, mydir):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        my_file = Path(
            f"{settings.STATIC_UPLOAD_DIR}/{mydir}/{uploaded_file.name}")
        if not my_file.is_file():
            save_file(uploaded_file, mydir)
            return redirect(f"/chat/{mydir}/")
        else:
            return JsonResponse({'success': False, 'data': "The File Existers allready"})
    else:
        form = FileUploadForm()
    return render(request, 'save-static.html', {'form': form})
