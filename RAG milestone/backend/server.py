from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
from controller import ControllerFactory  
from db import VectorDBFactory
from config import get_settings 
from components import ChatModelFactory
import pandas as pd 
from starlette.middleware.cors import CORSMiddleware
from logger import get_logger

logger = get_logger(__name__)

# Load the CSV file into a list
def load_csv_to_dataframe(file_path: str) -> list:
    try:
        df = pd.read_csv(file_path)
        return df["content"].values.tolist()
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    vdb_factory = VectorDBFactory(config=settings)
    vdb = vdb_factory.create_vector_db(db_type=settings.DB_TYPE)

    chat_factory = ChatModelFactory(config=settings)
    
    controller_factory = ControllerFactory(config=settings)
    rag_controller = controller_factory.create_controller(controller_type=settings.CONTROLLER_TYPE, vdb=vdb, chat_factory=chat_factory)
    rag = rag_controller.initialize(texts=load_csv_to_dataframe(settings.DATA_PATH))

    app.state.rag = rag

    logger.info("RAG model initialized successfully.")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
settings = get_settings()



@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    question = data.get("question")
    logger.debug(f"Received question: {question}")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    rag = app.state.rag
    return StreamingResponse(rag.stream(question), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) 