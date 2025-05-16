from .ChromaVectorDB import ChromaVectorDB
from .VectorDB import VectorDB
from enums import VDBEnums
from logger import get_logger

logger = get_logger(__name__)

class VectorDBFactory:
    """Factory class for creating different vector database instances."""
    def __init__(self, config: dict):
        self.config = config

    def create_vector_db(self, db_type: str) -> VectorDB:
        """
        Creates a vector database instance based on the provided type.

        Args:
        - db_type (str): The type of vector database (e.g., "chroma").

        Returns:
        - VectorDB: An instance of the specified vector database.

        Raises:
        - ValueError: If an unsupported database type is requested.
        """

        if db_type == VDBEnums.CHROMA_DB.value:
            logger.info("Creating Chroma VectorDB")
            return ChromaVectorDB(
                embedding_model=self.config.EMBEDDINGS_MODEL_NAME, 
                persist_directory=self.config.PERSIST_DIRECTORY
            )
        raise ValueError(f"Unsupported database type: {db_type}. Available options: {list(VDBEnums.__members__.keys())}")