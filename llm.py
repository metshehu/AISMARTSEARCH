import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file

# Now you can access the variables like this:
# "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" #
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
print(respones)
