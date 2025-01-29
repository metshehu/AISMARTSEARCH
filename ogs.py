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


# target_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads')
path('Asking/<str:mydir>/<str:text>', views.asking, name='Asking'),
# os.makedirs(target_dir, exist_ok=True)


def context_aware_responses_og(query, Question_history, Answer_history, data, user):
    # openai.api_key = settings.OPENAI_K
    temp = gettemp(user)
    client = OpenAI(api_key=settings.OPENAI_KEY)
    print(query, "this is querry", "-"*100)
    if (len(data) == 0):
        return "The context does not contain sufficient information to answer this question.--2"

    messages = [
        {"role": "system", "content": (
            "You are a context-aware search engine. You must return responses **only** based on the provided context  and the user's past interactions. "
            "You are not allowed to infer or assume answers if the context does not provide sufficient information. "
            "If the context does not contain sufficient information to answer the question, respond only with: "
            "'The context does not contain sufficient information to answer this question.' "
            "Do not provide additional information, guesses, or speculative answers."
        )},
        {"role": "user", "content": f"current Question: {query}"},
    ]

    # {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {  # remove the cuhnks cus it allred add
    #    query}\n\n Past Questino:{Question_history}\n\n past Answer:{Answer_history}"},
    # past quest/ answer make it add context insted of dumping data

    print('-'*100)
    print(len(data))
    print('<>'*50)
    for i in data:
        print(data[i]['chunks'])
        print(len(data[i]['chunks']))
        print(data[i]['score'])
    print('-'*100)

    addHistory(Question_history, Answer_history, messages)

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


def addContextog(data, messages):
    for i in data:
        print(i+'-'*20)
        text = ''
        for j in data[i]['chunks']:
            text += f"```text {j}```"

            newdic = {"role": "system", "content": f"The following file: '{
                i}'contains the following relevant information to support the answer:```text{text} ```"}

    # {"role": "system", "content": "The following file": "Computer science and engineering sylabus” contains the following relevant information to support the answer:"}
    #        message.append(newdic)

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



def addHistroy(question_history, answer_histry, message):
    index = 0
    question_history = question_history[-10:]
    answer_histry = answer_histry[-10:]

    for q, a in zip(question_history, answer_histry):
        question_dic = {
        "role": "user", "content": f" this is a past question number {index} {q}"}
        answer_dic = {"role": "assistant",
                  "content": f" this is a past answer for question number {index} {a}"}

        message.append(question_dic)
        message.append(answer_dic)
        index += 1

def asking(request, mydir, text):
    fileEmbedings = Parsers(settings.OPENAI_KEY)
    query_vector = fileEmbedings.embedquerry(text)
    chunks, vectors = system_file_parser(query_vector, mydir)
    history = user_history(mydir)
    res = context_aware_responses(chunks, text)
    return HttpResponse(f"{res}")

