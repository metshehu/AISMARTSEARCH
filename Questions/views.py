import os
from os import walk
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI

from Main import Parsers

from .forms import FileUploadForm

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
    ("system", "your a Amarican RedNeck"),
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


def system_file_parser(querry_vector, mydir):
    mypath = settings.STATIC_UPLOAD_DIR+'/'+mydir+'/'
    parser = Parsers(settings.OPENAI_KEY)
    vectorlist = []
    chunkslist = []
    files = allFileformat(mypath, '.csv')
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        closest_index = parser.cosine_search(vectors, querry_vector)
        top3 = parser.cosine_search_top3(vectors, querry_vector, 80)
        for j in top3:
            chunkslist.append(chunks[j])
            vectorlist.append(vectors[j])
    return (chunkslist, vectorlist)


def context_aware_responses(chunks, query):
    # openai.api_key = settings.OPENAI_K
    client = OpenAI(api_key=settings.OPENAI_KEY)
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "your are a context aware search that runes the valid respose baste on the context given"},
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


def front(request, mydir):

    a = getalldirs(settings.STATIC_UPLOAD_DIR)
    l = []
    for i in a:
        l.append(allFileformat(settings.STATIC_UPLOAD_DIR+'/'+i, '.pdf'))
    b = allFileformat(settings.STATIC_UPLOAD_DIR+'/'+mydir, '.csv')
    c = allFileformat(settings.STATIC_UPLOAD_DIR+'/'+mydir, '.pdf')
    context = {
        'list': a,
        'filescsv': b,
        'filespdf': l
    }
    return render(request, 'home.html', context)


def asking(request, mydir, text):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)
    chunks, vectors = system_file_parser(query_vector, mydir)
    res = context_aware_responses(chunks, text)
    return HttpResponse(f"{res}")


def asking_normal(mydir, text):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)
    chunks, vectors = system_file_parser(query_vector, mydir)
    res = context_aware_responses(chunks, text)
    return res


def chat(request, dir):
    qes = ""
    mypath = settings.STATIC_UPLOAD_DIR+'/'+dir
    if (request.method == "POST"):
        text = request.POST.get("question")
        qes = asking_normal(dir, text)

    files = allFileformat(mypath, '.pdf')
    context = {
        'dir': dir,
        'answer': qes,
        'files': files

    }
    return render(request, 'chat.html', context)


def home(request):
    mypath = settings.STATIC_UPLOAD_DIR
    b = getalldirs(mypath)
    context = {
        'list': b
    }
    return render(request, 'home.html', context)


def makedir(request, dir_name):
    mypath = settings.STATIC_UPLOAD_DIR
    b = getalldirs(mypath)

    target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads', dir_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Directory {dir_name} created at {target_dir}")
    else:
        print(f"Directory {dir_name} already exists at {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
    return JsonResponse({'list': b})
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
            return redirect(f"/hello/chat/{mydir}/")
        else:
            return JsonResponse({'success': False, 'data': "The File Existers allready"})
    else:
        form = FileUploadForm()
    return render(request, 'save-static.html', {'form': form})
