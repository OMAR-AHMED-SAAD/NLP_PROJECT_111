from enums import ControllerEnums
from .RAGController import RAGController
from .BaseController import BaseController
from db import VectorDB
from components import ChatModelFactory


class ControllerFactory:
    """Factory class for creating different controller instances."""

    def __init__(self, config: dict):
        '''
        Initializes the ControllerFactory with a configuration.
        
        Args:
        - config (dict): The configuration
        '''
        self.config = config

    def create_controller(self, controller_type: str, vdb: VectorDB,
                          chat_factory: ChatModelFactory) -> BaseController:
        """
        Creates a controller instance based on the provided type.

        Args:
        - controller_type (str): The type of controller (e.g., "AgenticRAGController").

        Returns:
        - BaseController: An instance of the specified controller.

        Raises:
        - ValueError: If the specified controller type is not supported.
        """

        if controller_type == ControllerEnums.RAG_CONTROLLER.value:
            return RAGController(
                embeddings_model_name=self.config.EMBEDDINGS_MODEL_NAME,
                system_prompt=self.config.SYSTEM_PROMPT,
                k=self.config.K,
                memory_size=self.config.MEMORY_SIZE,
                vdb=vdb,
                chat_factory=chat_factory,
                ingest = self.config.INGEST)

        raise ValueError(
            f"Unsupported controller type: {controller_type}. Available options: {list(ControllerEnums.__members__.keys())}"
        )
