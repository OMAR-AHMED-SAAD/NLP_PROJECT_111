from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class VectorDB(ABC):

    @abstractmethod
    def ingest(self, texts: List[str]) -> None:
        '''
        Ingests a list of texts into the vector database after converting them to Documents.

        args:
        - texts (list[str]): A list of texts to ingest
        '''
        pass

    @abstractmethod
    def query(self, text: str, k: int) -> List[str]:
        '''
        Queries the vector database for the most similar texts to the input text.

        args:
        - text (str): The input text
        - k (int): The number of similar texts to return

        returns:
        - list[str]: The most similar texts
        '''
        pass

    @abstractmethod
    def update(self, texts: List[str]) -> None:
        '''
        Updates the vector database with the most recent text.

        args:
        - texts (list[str]): New text to update the database with
        '''
        pass

    def _convert_to_document(self, texts: List[str]) -> List[Document]:
        '''
        Converts a list of texts into a list of documents.

        args:
        - texts (list[str]): A list of texts

        returns:
        - list[Document]: A list of documents
        '''

        return [Document(page_content=text) for text in texts]
    
    @abstractmethod
    def drop(self) -> None:
        '''
        Drops the vector database.
        '''
        pass