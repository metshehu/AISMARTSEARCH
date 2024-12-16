import csv
import os

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)


def SaveVector(VectorEmbedList):
    with open('vector.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for value in VectorEmbedList:
            writer.writerow([value])


def PandasSave(VectorEmbedQ):
    df = pd.DataFrame([VectorEmbedQ])
    df.to_csv('vector2.csv', index=False, header=False)


def ReadFromFile(name):
    df = pd.read_csv(name, header=None)
    vector_from_csv = df.values.flatten().tolist()
    return vector_from_csv


class Parsers():
    def __init__(self, apikey, file, chunks, overlap):
        self.apikey = apikey
        self.file = file
        self.chunks = chunks
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=150, length_function=len
        )
        self.embedings = []

        # self.spliter=spltier#add later

    def setFile(self, file):
        self.file = file

    def Vectoraiz(self):
        pages = PyPDFLoader(self.file).load()
        docs = self.splitter.split_documents(pages)
        self.embedings = Chroma.from_documents(docs, OpenAIEmbeddings())
        return self.embedings

    def querry(self, question: str):
        vectorEmbedQuery = OpenAIEmbeddings().embed_query(question)
        answer = self.embedings.similarity_search_by_vector(vectorEmbedQuery)
        return answer


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Load a PDF document
    loader = PyPDFLoader("01_BIA_Njoftimi me Lenden - Syllabusi.pdf")
    pages = loader.load()
    # text_splitter = CharacterTextSplitter(
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150, length_function=len
    )
    docs = text_splitter.split_documents(pages)
    db = Chroma.from_documents(docs, OpenAIEmbeddings())

    query = "aritjeet"
    vectorEmbedQuery = OpenAIEmbeddings().embed_query(query)
    answer = db.similarity_search_by_vector(vectorEmbedQuery)

    # for i in answer:
    # print(i.page_content)
    # print("-----------------")

    PandasSave(vectorEmbedQuery)
    a = ReadFromFile('vector2.csv')
    for i in db.similarity_search_by_vector(a):
        print(i.page_content)
    print("-----------------")

# print(a)
# If it's a list or numpy array
