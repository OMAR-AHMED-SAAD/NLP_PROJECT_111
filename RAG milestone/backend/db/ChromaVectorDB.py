from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .VectorDB import VectorDB
from typing import List
from logger import get_logger

logger = get_logger(__name__)

class ChromaVectorDB(VectorDB):

    def __init__(self, embedding_model: str, persist_directory="./chroma_db"):
        '''
        Initializes the ChromaVectorDB with an embedding model and a persist directory.
        
        Args:
        - embedding_model (str): The embedding model
        - persist_directory (str): The persist directory
        '''
        super().__init__()
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_directory = persist_directory
        self.chroma_db = Chroma(persist_directory=self.persist_directory,
                                embedding_function=self.embedding)

    def ingest(self, texts: List[str]):
        '''
        Ingests a list of texts into the vector database after converting them to Document.

        args:
        - texts (list[str]): A list of texts to ingest
        '''
        docs = self._convert_to_document(texts)
        
        logger.info(f"Ingesting {len(docs)} documents into the ChromaDB.")
        self.chroma_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory)

    def query(self, text: str, k: int) -> List[str]:
        '''
        Queries the vector database for the most similar texts to the input text.

        args:
        - text (str): The input text
        - k (int): The number of similar texts to return

        returns:
        - list[str]: The most similar texts
        '''
        return self.chroma_db.similarity_search(
            query=text, k=k)


    def update(self, texts: List[str]) -> None:
        '''
        Updates the vector database with the most recent text.

        args:
        - texts (list[str]): New text to update the database with
        '''
        docs = self._convert_to_document(texts)
        self.chroma_db.add_documents(documents=docs, embedding=self.embedding)

    def drop(self):
        '''
        Drops the vector database.
        '''
        self.chroma_db.delete_collection()