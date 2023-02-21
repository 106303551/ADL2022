from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = { #建成一個token2idx dict 
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)}, #依token為key i為value
        }

    @property #使其不能被更改
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD] #用於pad_to_len的pad_id 呼叫會回傳其value(換個名字方便呼叫)

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK] #出現沒紀錄的token時回傳的unknown_id 呼叫會回傳其value(換個名字方便呼叫)

    @property
    def tokens(self) -> List[str]: #呼叫回傳token list
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int: #以str傳入token找id(回傳int)
        return self.token2idx.get(token, self.unk_id) #回傳id(key),如token不存在回傳unk_id(1)

    def encode(self, tokens: List[str]) -> List[int]: #以list[str]傳入token找id(回傳list[int])
        return [self.token_to_id(token) for token in tokens]

    def encode_batch( #分批次以list[str]傳入token找id(批次回傳list[int])
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens] #此處tokens為一list 因此呼叫encode
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len #紀錄最長的ids list的length
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id) #使所有ids長度一樣
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:#拉長ids
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs] #加長ids的長度至最常ids的length 使所有ids長度一樣
    return paddeds