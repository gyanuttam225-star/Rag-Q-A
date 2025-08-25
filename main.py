from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, shutil

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

open_api_key = os.getenv("OPEN_API_KEY")

app = FastAPI()

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

UPLOAD_DIR = "uploads"
INDEX_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

db = None


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global db

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)  # persist index

    return {"message": f"Uploaded and processed {file.filename}"}


@app.post("/chat/")
async def chat(query: str = Form(...)):
    global db

    if db is None:
        if os.path.exists(INDEX_DIR):
            embeddings = OpenAIEmbeddings()
            db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        else:
            return JSONResponse(content={"error": "No PDF uploaded yet"}, status_code=400)

    docs = db.similarity_search(query, k=3)

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key)
    context = "\n".join([d.page_content for d in docs])
    response = llm.predict(f"Answer the question based on context:\n{context}\n\nQuestion: {query}")

    return {"answer": response}
