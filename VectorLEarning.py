from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

endpoint = os.getenv("OPENAI_API_KEY")
raw_documents = TextLoader('DocumentTest.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "tell me about paris"
docs = db.similarity_search(query)
print(docs[0].page_content)
print("-----------------")
embanding=OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embanding)
print(docs[0].page_content)

