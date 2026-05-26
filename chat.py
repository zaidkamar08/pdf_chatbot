import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

print("Loading PDF...")
docs = PyMuPDFLoader("notes.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)

print("Building knowledge base (takes 1-2 mins first time)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(model="llama-3.1-8b-instant")
print("\nReady! Type your question or 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    pages = list(set([str(d.metadata.get('page','?')) for d in docs]))
    response = llm.invoke([
        SystemMessage(content=f"Answer using only this context. If unsure say 'I don't know'.\n\n{context}"),
        HumanMessage(content=question)
    ])
    print(f"\nAnswer: {response.content}")
    print(f"Pages: {', '.join(pages)}\n")