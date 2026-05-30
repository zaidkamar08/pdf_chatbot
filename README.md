# PDF Chatbot

Ask questions about any PDF using AI. Built with Python, Groq (LLaMA 3.1), and FAISS.

## Live Demo

Try it here: https://pdf-chatbot-extractor.streamlit.app

Upload any PDF and start asking questions instantly. No setup needed.

## What it does

- Upload any PDF through the browser
- Ask questions in plain English
- Get accurate answers with page numbers
- Powered by LLaMA 3.1 running on Groq

## Tech used

- Groq API (free) — LLaMA 3.1 8b — the AI brain that generates answers
- FastEmbed — converts text to vectors for semantic search
- FAISS — stores and searches vectors to find relevant chunks
- LangChain — connects all the components together
- Streamlit — web interface

## How it works

1. Your PDF is split into 500 character chunks
2. Each chunk is converted into a vector (list of numbers representing meaning)
3. When you ask a question, it is also converted to a vector
4. FAISS finds the 3 chunks most similar to your question
5. Those chunks are sent to the LLaMA model as context
6. The model answers using only that context and returns page numbers

## Run locally

1. Clone the repo
2. Create a `.env` file with your `GROQ_API_KEY` from console.groq.com
3. Install dependencies: `pip install -r requirements.txt`
4. Run the terminal version: `python chat.py`
5. Or run the web UI: `streamlit run app.py`
6. Open browser at `http://localhost:8501`

## What I learned

- How RAG (Retrieval Augmented Generation) works end to end
- Why chunk size matters for answer quality and how to tune it
- How text gets converted to vectors for semantic search
- How to connect an LLM to custom data without training or fine-tuning
- How to deploy a Python AI app to the web for free
