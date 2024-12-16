import ast
import csv
import os

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (CharacterTextSplitter,
                                      MarkdownTextSplitter,
                                      RecursiveCharacterTextSplitter,
                                      TokenTextSplitter)
from sklearn.metrics.pairwise import cosine_similarity


def SaveVector(VectorEmbedList):
    with open('vector.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for value in VectorEmbedList:
            writer.writerow([value])


def ReadFromFile(file_path):
    df = pd.read_csv(file_path, header=None)
    vector_from_csv = df.values.flatten().tolist()
    return vector_from_csv


class Parsers():
    def __init__(self, apikey, spliter):
        self.apikey = apikey
        self.vectors = OpenAIEmbeddings(openai_api_key=apikey)
        self.splitter = spliter        # self.spliter=spltier#add later
        self.chunks = []
        self.vectordata = []
        self.querry = []
        self.buffer = []
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.apikey, model="text-embedding-3-large")

    def SetSpliter(self, spliter, chuncksize, overlap):
        match spliter:
            case "Character":
                self.splitter = CharacterTextSplitter(
                    chunk_size=chuncksize, chunk_overlap=overlap)
            case "RecursiveCharacter":
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chuncksize, chunk_overlap=overlap)
            case "Token":
                self.splitter = TokenTextSplitter(
                    chunk_size=chuncksize, chunk_overlap=overlap)
            case "Markdown":
                self.splitter = MarkdownTextSplitter(
                    chunk_size=chuncksize, chunk_overlap=overlap)
            case _:
                self.splitter = CharacterTextSplitter(
                    chunk_size=chuncksize, chunk_overlap=overlap)

#
#    def SaveCsv(self, file_path):
#        df = pd.DataFrame([self.db._collection.get()['vectors']])
#        print(self.db._collection.get())
#        df.to_csv(file_path+'.csv', index=False, header=False)
#
#    def setFile(self, file_path):
#        self.file_path = file_path
#
#    def Load(self, file_path, file_type):
#        if (file_type == 'png'):
#            self.file_path = PyPDFLoader()

    # curl https://api.openai.com/v1/vectors \
#  -H "Content-Type: application/json" \
#  -H "Authorization: Bearer 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" \
#  -d '{
#    "input": "this is a randomg text i dont give a fuck llaglaglgallgglaagllgaagl",
#    "model": "text-embedding-3-small"
#  }'

    def Print(self, showList):
        for vector in showList:
            print(str(vector)[:100]+'top')
            print(str(vector)[len(showList)-100:]+'bottem')

    def embedd(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        self.chunks = self.splitter.split_documents(documents)
        # self.chunks = [str(i) for i in self.chunks]
        self.chunks = [doc.page_content for doc in self.chunks]
        self.vectors = self.embeddings.embed_documents(self.chunks)
        return self.vectors

    def getChunks(self):
        return self.chunks

    def embedquerry(self, querry):
        self.querry = self.embeddings.embed_query(querry)
        return self.querry

    def SaveCsv(self, file_path, name, vectors, chunks):
        df = pd.DataFrame({
            "chunks": chunks,
            "vectors": vectors
        })
        if not file_path.endswith("/"):
            file_path += "/"
        locat = file_path+name+'.csv'
        df.to_csv(locat, index=False)

    def ReadFromFile(self, file_path):
        df = pd.read_csv(file_path)
        chunks = df["chunks"].tolist()
#        vectors = df["vectors"].tolist()
        vectors = df["vectors"].apply(
            ast.literal_eval).tolist()  # Convert strings to lists
        return (chunks, vectors)

    def cosine_search(self, vectors, query_vector):
        vectors = np.array(vectors)
        query_vector = np.array(query_vector)

        query_vector = query_vector.reshape(1, -1)
        distances = cosine_similarity(vectors, query_vector)

        closest_index = np.argmax(distances)
        return closest_index

    def cosine_search_chunks(self, data, query_vector):
        chunks = data[0]
        vectors = np.array(data[1])
        query_vector = np.array(query_vector)
        query_vector = query_vector.reshape(1, -1)
        distances = cosine_similarity(vectors, query_vector)
        closest_index = np.argmax(distances)
        return chunks[closest_index]


# ------------------------------------------------------------------------------------------------
# Step 8: Save the DataFrame to a CSV file_contet


    def Vectoraiz(self, file_path):
        self.file_contet = PyPDFLoader(file_path).load()
        docs = self.splitter.split_documents(self.file_contet)
        self.db = Chroma.from_documents(docs, OpenAIEmbeddings())
        return self.db

    def querry(self, question: str):
        vectorEmbedQuery = OpenAIEmbeddings().embed_query(question)
        answer = self.db.similarity_search_by_vector(vectorEmbedQuery)
        return answer


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Load a PDF document
    loader = PyPDFLoader("01_BIA_Njoftimi me Lenden - Syllabusi.pdf")
    file_contet = loader.load()
    # text_splitter = CharacterTextSplitter(
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150, length_function=len
    )
    docs = text_splitter.split_documents(file_contet)
    db = Chroma.from_documents(docs, OpenAIEmbeddings())

    query = "importen things"
    vectorEmbedQuery = OpenAIEmbeddings().embed_query(query)
    answer = db.similarity_search_by_vector(vectorEmbedQuery)
#    print(vectorEmbedQuery)

    a = Parsers(os.getenv("OPENAI_API_KEY"),
                "01_BIA_Njoftimi me Lenden - Syllabusi.pdf", text_splitter)
    a.Vectoraiz()
    answer2 = a.querry(query)
    # for i in answer:
    # for i in answer:
    # for i in answer:
    # print(i.page_content)
    # print("-----------------")

    # PandasSave(vectorEmbedQuery)
    # a = ReadFromFile('vector2.csv')
    # for i in answer2:
    #    print(i.page_content)
    #    print('1')
    # print("-----------------")

# print(a)
# If it's a list or numpy array