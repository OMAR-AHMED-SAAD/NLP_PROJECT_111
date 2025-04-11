import torch 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union
from tokenizers import Tokenizer
from .bpe_tokenizer import BPETokenizer

class QADataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: BPETokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        question = sample["question"]
        answer = sample["answer"]
        context = sample["context"]

        # Tokenize the question and answer
        self.tokenizer.set_max_length(25)
        question_tokens, attention_mask_question = self.encode(question, self.tokenizer.get_tokenizer())
        self.tokenizer.set_max_length(10)
        answer_tokens, attention_mask_answer = self.encode(answer, self.tokenizer.get_tokenizer())

        return {
            "question": torch.tensor(question_tokens, dtype=torch.long),
            "attention_mask_question": torch.tensor(attention_mask_question, dtype=torch.long),
            "answer": torch.tensor(answer_tokens, dtype=torch.long),
            "attention_mask_answer": torch.tensor(attention_mask_answer, dtype=torch.long)
        }
    
    def encode(self, text: str, tokenizer: Tokenizer) -> Tuple[List[int], List[int]]:
        '''
        Encode a single text into tokens.
        
        Args:
            text (str): The text to encode.
            tokenizer (Tokenizer): The tokenizer to use.
        '''
        tokens = tokenizer.encode(text)
        attention_mask = [1 if tok != tokenizer.token_to_id("[PAD]") else 0 for tok in tokens.ids]
        return tokens.ids, attention_mask


    
