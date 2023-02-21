from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab,pad_to_len

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        input=torch.tensor([data['tokens'].tolist() for data in samples])#list[tensor] ok
        label=[data['tags']for data in samples] #ok
        len_list=[len(data)for data in label] #ok
        tags=pad_to_len(label,self.max_len,0) #ok
        tags=torch.tensor(tags)
        id=[data['id'] for data in samples] #ok
        return {'tokens':input,'tags':tags,'id':id,'len':len_list} #ok

