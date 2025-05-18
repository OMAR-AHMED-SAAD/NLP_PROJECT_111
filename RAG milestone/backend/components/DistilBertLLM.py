import re
from langchain.llms.base import LLM
from typing import Optional, List
from .DistilBert import DistilBert
from pydantic import PrivateAttr
from logger import get_logger


logger = get_logger(__name__)

class DistilBertLLM(LLM):
    _model: DistilBert = PrivateAttr()  

    class Config:
        arbitrary_types_allowed = True 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = DistilBert()

    @property
    def _llm_type(self) -> str:
        return "distilbert-qa"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:
        
        # logger.debug(f"DistilBertLLM: _call with prompt: {prompt}")
        ctx = re.search(r"<context>(.*)</context>", prompt, re.DOTALL)
        context = ctx.group(1).strip() if ctx else ""

        # pull out memory
        mem = re.search(r"<memory>(.*)</memory>", prompt, re.DOTALL)
        memory = mem.group(1).strip() if mem else ""


        question = re.search(r"<question>(.*)</question>", prompt, re.DOTALL)
        question = question.group(1).strip() if question else ""
        logger.debug(f"DistilBertLLM: _call with question: {question}")

        context_memory = context + "\n" + memory

        return self._model.answer_question(question=question, context=context_memory)
