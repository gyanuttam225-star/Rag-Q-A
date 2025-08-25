from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
open_api_key=os.getenv("OPEN_API_KEY")
app = FastAPI()
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# memory for storing FAISS index
db = None

@app.get("/")
async def root():
    return {"message": "PDF RAG API is running! Use /upload_pdf/ to upload and /chat/ to query."}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global db

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # embeddings
    embeddings = OpenAIEmbeddings()

    # create FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    return {"message": f"Uploaded and processed {file.filename}"}


@app.post("/chat/")
async def chat(query: str = Form(...)):
    global db
    if db is None:
        return JSONResponse(content={"error": "No PDF uploaded yet"}, status_code=400)

    # retrieve relevant chunks
    docs = db.similarity_search(query, k=3)

    # pass to LLM for answering
    llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=open_api_key)
    context = "\n".join([d.page_content for d in docs])
    response = llm.predict(f"Answer the question based on context:\n{context}\n\nQuestion: {query}")

    return {"answer": response}
