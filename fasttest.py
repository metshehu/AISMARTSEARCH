import os
from ast import literal_eval

import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)
from sklearn.metrics.pairwise import cosine_similarity

from Main import Parsers

if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150, length_function=len
    )

    TestingParser = Parsers(os.getenv("OPENAI_API_KEY"),
                            "/Users/metshehu/Documents/workingprojects/Python/MetiUseCase/01_BIA_Njoftimi me Lenden - Syllabusi.pdf", spliter)
#    TestingParser.Vectoraiz()
#    TestingParser.SaveCsv("this_is_a_test")
#    print(TestingParser.ReadFromFile("this_is_a_test.csv"))
#    TestingParser.embedd()
#    TestingParser.SaveCsv(
 #       "/Users/metshehu/Documents/workingprojects/Python/MetiUseCase/", 'othertest')
    tup = TestingParser.ReadFromFile(
        "/Users/metshehu/Documents/workingprojects/Python/MetiUseCase/static/uploads/01_BIA_Njoftimi me Lenden - Syllabusi.csv")
    othertup = TestingParser.ReadFromFile(
        "/Users/metshehu/Documents/workingprojects/Python/MetiUseCase/whatever.csv")
    mypath = "/Users/metshehu/Documents/workingprojects/Python/MetiUseCase/static/uploads/"
   print("[]"*10)
    for i in f:
        if(i[-4:]=='.pdf'):
            print(i)
    print("[]"*10)

    question = othertup[0]
    query_vector = othertup[1]
    chunks = tup[0]
    vectors = tup[1]
    print(question)

    vectors = np.array(vectors)
    query_vector = np.array(query_vector)

    id = TestingParser.cosine_search(vectors, query_vector)
    print(chunks[id])
    # print(query_vector.shape, "- this is query_vector")
    # print(vectors.shape, "- this is vector")
#    # vectors = vectors.reshape(1, -1)
    # query_vector = query_vector.reshape(1, -1)
    # # vectors = vectors.reshape(1, -1)
    # print(query_vector.shape, "- new query_vector")
    # print(vectors.shape, "- new vector")
    # # vectors = vectors.reshape(-1, 3072)

    # print("-"*10)
#    numpy_array = np.array(query_vector)
    # print(chunks[0])
    # print(numpy_array)

    # parsed_vectors = [literal_eval(vec) for vec in data]

    # print(type(query_vector[0]))
    # numpy_array = np.array(parsed_vectors)
    # print(query_vector)
    # distances = similarities = cosine_similarity(vectors, query_vector)
    # closest_index = np.argmax(distances)
    # closest_vector = chunks[closest_index]
    # print(closest_vector)
    # print("-"*10)
    # for i in question:
    # print(i)
    # print("-"*10)
    # print(distances)
    # # for i in range(len(tup[0])):
    # # print(str(tup[0][i])[:100]+'chunk')
    #    print(len(tup[1][i]))
    # print(str(tup[1][i])[:100]+'value')
    # TestingParser.SaveCsv('this_is_a_test')
    # TestingParser.ReadFromFile("this_is_a_test.csv")

    #    for i in TestingParser.querry("aritjeet"):
    #        print(i.page_content)
