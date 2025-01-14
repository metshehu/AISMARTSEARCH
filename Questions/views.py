import collections
import os
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

from .forms import FileUploadForm, MakeDirForm
from .models import History

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
    parser = Parsers(settings.OPENAI_KEY)
    vectorlist = []
    chunkslist = []
    files = allFileformat(mypath, '.csv')
    files_data = {}
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        # closest_index = parser.cosine_search(vectors, querry_vector)
        top3, similariti_score = parser.cosine_search_top3(
            vectors, querry_vector, 80)
        for j in top3:
            chunkslist.append(chunks[j])
            vectorlist.append(vectors[j])
        addfiledata(files_data, i, chunkslist, vectorlist, similariti_score)

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


def context_aware_responses(chunks, query, Question_history, Answer_history, data):
    # openai.api_key = settings.OPENAI_K
    client = OpenAI(api_key=settings.OPENAI_KEY)
    messages = [
        # {"role": "system", "content": "You are answers helpful assistant."},
        {"role": "system", "content": "your are answers context aware search that retruns the valid respose baste on the context given baste on the Question and use users Past Question and Answers ans a sorce as well to get context"},
        {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {
            query}\n\n Past Questino:{Question_history}\n\n past Answer:{Answer_history}"},

    ]
    addContext(data, messages)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150  # test on the higer end to the lowest 50-600
    )
    response_message = response.choices[0].message.content
    return response_message
    # return response['choices'][0]['message']['content'].strip()


def front(request, mydir):

    answers = getalldirs(settings.STATIC_UPLOAD_DIR)
    l = []
    for i in answers:
        l.append(allFileformat(settings.STATIC_UPLOAD_DIR+'/'+i, '.pdf'))
    b = allFileformat(settings.STATIC_UPLOAD_DIR+'/'+mydir, '.csv')
    c = allFileformat(settings.STATIC_UPLOAD_DIR+'/'+mydir, '.pdf')
    context = {
        'list': answers,
        'filescsv': b,
        'filespdf': l
    }
    return render(request, 'home.html', context)


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


def asking_normal(mydir, text):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)
    chunks, vectors, all_data = system_file_parser(query_vector, mydir)
    history = user_history(mydir)
    pastQuestion, pastAnswe = unpack_history(history)
    res = context_aware_responses(
        chunks, text, pastQuestion, pastAnswe, all_data)
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

            photoname = dirname+photo.name[-4:]
            uploadphoto(photoname, photo)
            return redirect(f"/chat/{dirname}/")
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
