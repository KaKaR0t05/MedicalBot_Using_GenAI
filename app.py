from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embidding
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import re

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


embeddings = download_hugging_face_embidding()

index_name = "medibot"

dbsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = dbsearch.as_retriever(search_type = "similarity", search_kwargs = {"k" : 3})

llm = OllamaLLM(model = "deepseek-r1:1.5b",
                configurable = {"temperature": 0.6, "num_predicts": 1000})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qna_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,qna_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods = ["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    
    raw_answer = response.get("answer", "")

# Use regex to remove the <think> block
    clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    print("Response: ", clean_answer)
    return clean_answer


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8080, debug = True)