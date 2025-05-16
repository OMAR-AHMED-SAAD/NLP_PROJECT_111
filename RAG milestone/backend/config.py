from pydantic_settings import BaseSettings
from prompts import SYSTEM_PROMPT
class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str  # Required for Google Generative AI

    # Controller Settings
    CONTROLLER_TYPE: str  # Determines which controller to use (e.g., "RAGController")

    # VectorDB Settings
    DB_TYPE: str  # Type of vector database (e.g., "chroma_db")
    K: int = 1  # Number of top-k results to retrieve
    PERSIST_DIRECTORY: str  # Directory for persisting Chroma DB
    EMBEDDINGS_MODEL_NAME: str  # Name of the embeddings model

    DB_COLLECTION_NAME: str=None  # Name of the collection
    EMBEDDINGS_SIZE: int=1024  # Size of the embeddings

    GOOGLE_MAX_OUTPUT_TOKENS: int  # Maximum number of tokens to output for Google Generative AI
    # Embedding & Model Settings
    MODEL_CHAT_NAME: str  # Name of the main chat model
    MODEL_CHAT_CLASS: str  # Class of the main chat model

    # Prompts
    SYSTEM_PROMPT: str=SYSTEM_PROMPT  # System prompt for the chat model

    MEMORY_SIZE: int=10  # Size of the memory for the RAG model

    INGEST: bool = False  # Flag to indicate if the model should ingest data

    DATA_PATH: str = None  # Path to the data file (e.g., CSV file)

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    """Returns the application settings from environment variables."""
    return Settings()
