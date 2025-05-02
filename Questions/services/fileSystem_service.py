import collections
import os
import shutil
from itertools import chain
from os import walk
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage

from Questions.models import Chunk, History, UserValues

from .parser_service import Parsers


def getalldirs(mypath):
    user = []
    for dirpath, dirnames, filenames in walk(mypath):
        user = dirnames
        break
    return user


def allFileformat(mypath, format):
    files = []
    for dirpath, dirnames, filenames in walk(mypath):
        csv_files = [file for file in filenames if file.endswith(format)]
        files.extend(csv_files)
        break
    return files


def remove_csv(user):
    user_path = settings.STATIC_UPLOAD_DIR + f"/{user}"

    files = allFileformat(user_path, ".csv")

    for i in files:
        newpath = user_path + f"/{i}"
        if os.path.exists(newpath):
            os.remove(newpath)
            print(f"File '{newpath}' has been deleted.")
        else:
            print(f"File '{newpath}' does not exist.")


def delet_path(user):
    path = os.path.join(settings.STATIC_UPLOAD_DIR, user)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Directory '{path}' deleted successfully.")
    else:
        print(f"Directory '{path}' does not exist.")


def delet_photo(user):
    user_photos_path = os.path.join(settings.BASE_DIR, "static/userphotos", f"{user}.png")
    if os.path.exists(user_photos_path):
        os.remove(user_photos_path)
        print(f"File '{user_photos_path}' has been deleted.")
    else:
        print(f"File '{user_photos_path}' does not exist.")


def recrate_csvs(user_path, user, parser):
    pdf_files = allFileformat(user_path, ".pdf")
    word_files = allFileformat(user_path, ".docx")
    files = pdf_files + word_files
    for file_name in files:
        file_url = f"{settings.STATIC_UPLOAD_DIR}/{user}/{file_name}"
        fileChunks, fileEmbedings = parser.embedd(file_url)
        parser.SaveCsv(
            settings.STATIC_UPLOAD_DIR + "/" + user, file_name[:-4], fileEmbedings, fileChunks
        )


def reembedfiles(user):
    user_path = settings.STATIC_UPLOAD_DIR + f"/{user}"
    user_value = UserValues.objects.filter(user=user).first()
    parser = Parsers(settings.OPENAI_KEY)
    spliter = user_value.splitter
    chunksize = user_value.chunksize
    overlap = user_value.overlap
    parser.SetSpliter(spliter=spliter, chuncksize=chunksize, overlap=overlap)

    remove_csv(user)
    recrate_csvs(user_path, user, parser)


def addfiledata(dic, file_name, chunks, vectors, sim_score):
    dic[file_name] = {"chunks": chunks, "vectors": vectors, "score": sim_score}


def sort_data(files_data):
    sorted_files = sorted(files_data.items(), key=lambda item: item[1]["score"], reverse=True)
    top_10_files = sorted_files[:10]
    top_10_chunks = list(chain.from_iterable(item[1]["chunks"] for item in top_10_files))

    top_10_vectors = list(chain.from_iterable(item[1]["vectors"] for item in top_10_files))

    sorted_files_dict = collections.OrderedDict(top_10_files)

    return (top_10_chunks, top_10_vectors, sorted_files_dict)


def system_file_parser(querry_vector, user):
    mypath = settings.STATIC_UPLOAD_DIR + "/" + user + "/"
    parser = Parsers(settings.OPENAI_KEY)  # ✅

    vectorlist = []
    chunkslist = []
    files = allFileformat(mypath, ".csv")
    files_data = {}
    for i in files:
        chunks, vectors = parser.ReadFromFile(mypath + i)
        # closest_index = parser.cosine_search(vectors, querry_vector)
        top3, similariti_score = parser.cosine_search_top3(vectors, querry_vector, 30)
        for j in top3:
            chunkslist.append(chunks[j])
            vectorlist.append(vectors[j])
        if len(chunkslist) > 0:
            addfiledata(files_data, i, chunkslist, vectorlist, similariti_score)

            # print(files_data[i]['chunks'], files_data[i]['score'])

        chunkslist = []
        vectorlist = []

    top_10_chunks, top_10_vectors, sorted_files_dict = sort_data(files_data)

    return (top_10_chunks, top_10_vectors, sorted_files_dict)


def makedir(user_name):
    target_dir = os.path.join(settings.BASE_DIR, "static", "uploads", user_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Directory {user_name} created at {target_dir}")
    else:
        print(f"Directory {user_name} already exists at {target_dir}")
        os.makedirs(target_dir, exist_ok=True)


def save_file(uploaded_file, user):
    fs = FileSystemStorage(location=settings.STATIC_UPLOAD_DIR + f"/{user}")
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

    parser.SaveCsv(
        settings.STATIC_UPLOAD_DIR + "/" + user, uploaded_file.name, fileEmbedings, fileChunks
    )
    return fileEmbedings
