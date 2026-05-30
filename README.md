# PDF Chatbot

Ask questions about any PDF using AI. Built with Python, Groq (LLaMA 3.1), and FAISS.

## What it does
- Load any PDF
- Ask questions in plain English  
- Get answers with page numbers

## Tech used
- Groq API (free) — LLaMA 3.1 8b
- HuggingFace Embeddings — converts text to vectors
- FAISS — searches for relevant chunks
- LangChain — connects everything

## How to run
1. Clone the repo
2. Create a `.env` file with your `GROQ_API_KEY`
3. Add your PDF and rename it `notes.pdf`
4. Run `pip install -r requirements.txt`
5. Run `python chat.py`
6.   Open browser at http://localhost:8501
7. Upload any PDF and start asking questions

## What I learned
- How RAG (Retrieval Augmented Generation) works
- Why chunk size matters for answer quality
- How text gets converted to vectors for semantic search 
