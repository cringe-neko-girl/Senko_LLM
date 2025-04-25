from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from extractor import extract_text, extract_text_from_website
from trainer import train_model
from model import load_model, save_model, generate_reply
import shutil, os, json, socket
import uvicorn

app = FastAPI()
model = load_model()
DATA_PATH = 'data/knowledge_store.json'

class ChatRequest(BaseModel):
    query: str

class URLRequest(BaseModel):
    url: str

@app.get("/", response_class=HTMLResponse)
def upload_page():
    return open("intelli-model/uploader.html").read()

def load_knowledge():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r', encoding='utf8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Warning: knowledge store JSON is malformed. Starting fresh.")
    return []

def save_knowledge(knowledge_data):
    with open(DATA_PATH, 'w', encoding='utf8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=4)

def is_duplicate_entry(new_data, knowledge_data):
    # Check if the content is already present in the knowledge data (either by source or exact content)
    return any(entry["content"] == new_data["content"] for entry in knowledge_data)

@app.post("/upload")
async def upload(file: UploadFile):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    extracted_text = extract_text(file_path)
    os.remove(file_path)

    knowledge_data = load_knowledge()
    new_entry = {"source": file.filename, "content": extracted_text}

    # Avoid duplicates based on content
    if not is_duplicate_entry(new_entry, knowledge_data):
        knowledge_data.append(new_entry)
        train_model(extracted_text)
        save_model(model)
        save_knowledge(knowledge_data)
        return {"status": "Successfully trained on uploaded file"}
    else:
        return {"status": "This file's content has already been uploaded"}

@app.post("/upload-url")
async def upload_url(request: URLRequest):
    extracted_text = extract_text_from_website(request.url)

    if not extracted_text:
        return {"error": "Could not extract text from the given URL."}

    knowledge_data = load_knowledge()
    new_entry = {"source": request.url, "content": extracted_text}

    # Avoid duplicates based on content
    if not is_duplicate_entry(new_entry, knowledge_data):
        knowledge_data.append(new_entry)
        train_model(extracted_text)
        save_model(model)
        save_knowledge(knowledge_data)
        return {"status": f"Successfully trained on the content of {request.url}"}
    else:
        return {"status": "This URL's content has already been uploaded"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = generate_reply(request.query)
    return {"response": response}

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run():
    port = get_free_port()
    print(f"Running server on free port: {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    run()
