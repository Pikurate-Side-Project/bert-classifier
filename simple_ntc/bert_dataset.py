from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class TextClassificationCollator():

    def __init__(self, tokenizer: Any, max_length: int, with_text=True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
    
    def __call__(self, samples: List[Dict[str, str]]) -> Any:
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        
        # processing with huggingface tokenizer
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )

        return_value = {
            'inputs_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        if self.with_text:
            return_value['text'] = texts
        
        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts: str, labels: str) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item) -> Dict[str, str]:
        text = str(self.texts[item])
        label = str(self.labels[item])

        return {
            'text': text,
            'label': label
        }