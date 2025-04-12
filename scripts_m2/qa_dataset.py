import torch 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union
from tokenizers import Tokenizer
from .bpe_tokenizer import BPETokenizer

class QADataset(Dataset):
    def __init__(self, 
                 data: List[Dict[str, str]], 
                 tokenizer: BPETokenizer, 
                 set_padding: bool = False,
                 context_max_length: int=145,
                 answer_max_length: int=9,
                 question_max_length: int=25,
                 include_context: bool=False,
                 context_question_swap: bool=False) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.set_padding = set_padding
        self.answer_max_length = answer_max_length
        self.question_max_length = question_max_length
        self.include_context = include_context

        if self.include_context:
            self.context_question_swap = context_question_swap
            self.context_max_length = context_max_length
            self._filter_data(data)
            

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        question = sample["question"]
        answer = sample["answer"]
        context = sample["context"]
        answer_start = sample["answer_start"]
        answer_end = sample["answer_end"]

        # Tokenize the question and answer and context
        self.tokenizer.set_max_length(self.question_max_length)
        question_tokens, attention_mask_question, _ = self.encode(question, self.tokenizer.get_tokenizer())
        self.tokenizer.set_max_length(self.answer_max_length)
        answer_tokens, attention_mask_answer, _ = self.encode(answer, self.tokenizer.get_tokenizer())
        if self.include_context:
            self.tokenizer.set_max_length(self.context_max_length)
            context_tokens, attention_mask_context, context_offsets = self.encode(context, self.tokenizer.get_tokenizer())
            self.tokenizer.set_max_length(self.context_max_length + self.question_max_length + 1) # +1 for [SEP]
            if self.context_question_swap:
                context_question_tokens, attention_mask_context_question = self.encode_two_texts(question, context)
            else:
                context_question_tokens, attention_mask_context_question = self.encode_two_texts(context, question)
            start_idx, end_idx = self.prepare_start_end_indices(answer_start, answer_end, context_offsets)

        returned_data = {
            "question": torch.tensor(question_tokens, dtype=torch.long),
            "attention_mask_question": torch.tensor(attention_mask_question, dtype=torch.long),
            "answer": torch.tensor(answer_tokens, dtype=torch.long),
            "attention_mask_answer": torch.tensor(attention_mask_answer, dtype=torch.long),
        } 
        if self.include_context:
            returned_data["context"] = torch.tensor(context_tokens, dtype=torch.long)
            returned_data["attention_mask_context"] = torch.tensor(attention_mask_context, dtype=torch.long)
            returned_data["context_question"] = torch.tensor(context_question_tokens, dtype=torch.long)
            returned_data["attention_mask_context_question"] = torch.tensor(attention_mask_context_question, dtype=torch.long)
            returned_data["answer_start"] = torch.tensor(start_idx, dtype=torch.long)
            returned_data["answer_end"] = torch.tensor(end_idx, dtype=torch.long)

        return returned_data
    
    def encode(self, text: str, tokenizer: Tokenizer) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        '''
        Encode a single text into tokens.
        
        Args:
            text (str): The text to encode.
            tokenizer (Tokenizer): The tokenizer to use.
        '''
        tokens = tokenizer.encode(text)
        attention_mask = [1 if tok != tokenizer.token_to_id("[PAD]") else 0 for tok in tokens.ids]
        offsets = tokens.offsets
        return tokens.ids, attention_mask, offsets

    
    def encode_two_texts(self, text1: str, text2: str) -> Tuple[List[int], List[int]]:
        '''
        Encode two texts into tokens.
        
        Args:
            text1 (str): The first text to encode.
            text2 (str): The second text to encode.
            tokenizer (Tokenizer): The tokenizer to use.

        Returns:
            Tuple[List[int], List[int]]: The encoded tokens and attention mask.
        '''
        return self.tokenizer.encode_two_texts(text1, text2)
    
    def prepare_start_end_indices(self, 
                                  answer_start: int, 
                                  answer_end: int, 
                                  context_offsets: List[int]) -> Tuple[int, int]:
        '''
        Prepare the start and end indices for the answer in the context.
        
        Args:
            answer_start (int): The start index of the answer in the context.
            answer_end (int): The end index of the answer in the context.
            context_tokens (List[int]): The list of tokens in the context.
            tokenizer (BPETokenizer): The tokenizer to use.
            context (str): The context text.

        Returns:
            Tuple[int, int]: The start and end indices of the answer in the context tokens.
        '''
        start_idx = -1
        end_idx = -1
        # print(f"Answer start: {answer_start}, Answer end: {answer_end}")
        # print(f"Context offsets: {context_offsets}")
        for i, (curr_start_idx, curr_end_idx) in enumerate(context_offsets):
            if curr_start_idx <= answer_start <= curr_end_idx and start_idx == -1:
                start_idx = i
            if curr_start_idx <= answer_end <= curr_end_idx and end_idx == -1:
                end_idx = i
                break
        if end_idx == -1:
            end_idx = len(context_offsets) - 1
        if start_idx == -1:
            raise ValueError("Answer start index not found in context tokens.")
            
        return start_idx, end_idx
    
    def _filter_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter the data to remove samples that do not meet the length requirements.
        
        Args:
            data (List[Dict[str, str]]): The input data to filter.
        
        Returns:
            List[Dict[str, str]]: The filtered data.
        """
        filtered_data = []
        for sample in data:
            context = sample["context"]
            answer_start = sample["answer_start"]
            answer_end = sample["answer_end"]

            self.tokenizer.set_max_length(self.context_max_length)
            _, _, context_offsets = self.encode(context, self.tokenizer.get_tokenizer())

            try:
                _ = self.prepare_start_end_indices(answer_start, answer_end, context_offsets)
                filtered_data.append(sample)
            except ValueError:
                continue
        self.data = filtered_data
        print(f"Filtered dataset size: {len(self.data)} out of original {len(data)}")
    
