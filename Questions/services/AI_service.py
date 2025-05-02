from django.conf import settings
from openai import OpenAI

from Questions.models import Chunk, History, UserValues


def addContext(data, message):
    for file_name, content in data.items():
        # print(file_name + '-' * 20)

        chunks = "\n".join([f"```text\n{chunk}\n```" for chunk in content["chunks"]])

        newdic = {
            "role": "system",
            "content": (
                f"The following file: '{
                    file_name}' contains relevant information to support the answer: \n"
                f"{chunks}"
            ),
        }
        message.append(newdic)


def addHistory(question_history, answer_history, message):
    question_history = question_history[-10:]
    answer_history = answer_history[-10:]

    # Combine questions and answers into the message
    for index, (q, a) in enumerate(zip(question_history, answer_history)):
        question_entry = {"role": "user", "content": f"Past Question : {q}"}
        answer_entry = {"role": "assistant", "content": f"Past Answer : {a}"}
        message.append(question_entry)
        message.append(answer_entry)


def gettemp(user):
    return UserValues.objects.filter(user=user).first().temp


def context_aware_responses(query, Question_history, Answer_history, data, user):
    # openai.api_key = settings.OPENAI_K
    temp = gettemp(user)
    client = OpenAI(api_key=settings.OPENAI_KEY)
    # print(query, "this is querry", "-"*100)
    # if (len(data) == 0):
    #    return "The context does not contain sufficient information to answer this question.--2"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Agent designed to assist with answering questions based on the provided context. "
                "Your behavior and responses are governed by the following rules:\n\n"
                "1. **Context-Driven Responses Only**:\n"
                "- You must answer questions **only** based on the given context, user-provided information, or historical interactions.\n"
                "2. **No External Knowledge or Assumptions**:\n"
                "- You are not allowed to use knowledge outside the context, assume details, or provide speculative answers.\n\n"
                "3. **Clear and Concise Answers**:\n"
                "- Provide clear, accurate, and concise answers based on the available context.\n"
                "- Avoid verbose explanations unless explicitly requested.\n\n"
                "4. **Polite and Professional Tone**:\n"
                "- Maintain a polite and professional tone in all responses.\n\n"
                "5. **Error Handling**:\n"
                "- If you encounter ambiguous, contradictory, or invalid input, clarify the issue or state the limitations explicitly.\n\n"
                "You will now enter a question/answer session. Begin by addressing the user's query based on the context."
            ),
        },
        {"role": "user", "content": f"Current Question: {query}"},
    ]

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
