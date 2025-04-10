import torch 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union
from .bpe_tokenizer import BPETokenizer

class Phase1Dataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer_question: BPETokenizer, tokenizer_answer: BPETokenizer):
        self.data = data
        self.tokenizer_question = tokenizer_question
        self.tokenizer_answer = tokenizer_answer


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        question = sample["question"]
        answer = sample["answer"]

        # Tokenize the question and answer
        question_tokens, attention_mask_question = self.tokenizer_question.encode(question)
        answer_tokens, attention_mask_answer = self.tokenizer_answer.encode(answer)

        return {
            "question": torch.tensor(question_tokens, dtype=torch.long),
            "attention_mask_question": torch.tensor(attention_mask_question, dtype=torch.long),
            "answer": torch.tensor(answer_tokens, dtype=torch.long),
            "attention_mask_answer": torch.tensor(attention_mask_answer, dtype=torch.long)
        }
