import json
import logging
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import List, Dict
import torch
from tqdm.auto import tqdm
from utils import Vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

#建造單字表
def build_vocab(
    words: Counter, vocab_size: int, output_dir: Path, glove_path: Path
) -> None:
    common_words = {w for w, _ in words.most_common(vocab_size)}#紀錄前vocab_size個最常出現的單字 紀錄為一個set
    vocab = Vocab(common_words) #依common_words順序進入
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocab saved at {str(vocab_path.resolve())}")

    glove: Dict[str, List[float]] = {} #紀錄一個 glove 的 dict str為key list[float]為value
    logging.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path,encoding="utf-8") as fp:
        row1 = fp.readline() #讀一行
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1): #正則表達式
            # seek to 0
            fp.seek(0) #代0表重第一個字開始 檢查完後再跳回去初始位置
        # otherwise ignore the header

        for i, line in tqdm(enumerate(fp)): #line
            cols = line.rstrip().split(" ")#rstrip(刪除str末端的空格) split以空白格分開字
            word = cols[0] #紀錄此時的單字
            vector = [float(v) for v in cols[1:]] #跳過第一個(第一個為 word(單字))

            # skip word not in words if words are provided
            if word not in common_words: #跳過不需要的單字(沒在common_words)
                continue
            glove[word] = vector #記錄此word對應的vector
            glove_dim = len(vector) #紀錄vector長度

    assert all(len(v) == glove_dim for v in glove.values()) #檢查是否 所有glove內的value 長度都一樣 都回傳 True 如有false assert中斷跳error
    assert len(glove) <= vocab_size #確保glove內的vector不大於vocab_size(可能會少 glove內沒有此字)

    num_matched = sum([token in glove for token in vocab.tokens]) #計算有幾個vocab內的token被glove match到
    logging.info(
        f"Token covered: {num_matched} / {len(vocab.tokens)} = {num_matched / len(vocab.tokens)}"
    )
    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)]) #如果有token紀錄其glove vector 如果沒有此token(不在glove內)隨機賦予其vector
        for token in vocab.tokens
    ]
    embeddings = torch.tensor(embeddings) #換成tensor形式
    embedding_path = output_dir / "embeddings.pt" 
    torch.save(embeddings, str(embedding_path)) #紀錄tensor
    logging.info(f"Embedding shape: {embeddings.shape}") #size(6491,300)
    logging.info(f"Embedding saved at {str(embedding_path.resolve())}")


def main(args):
    seed(args.rand_seed)

    intents = set()  #設intents為一集合，不會有重複資料
    words = Counter() #建一計數器，可記錄words出現次數
    for split in ["train", "eval"]: #分別讀train and eval
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        intents.update({instance["intent"] for instance in dataset})#如果出現新的intents，update到集合
        words.update(
            [token for instance in dataset for token in instance["text"].split()] #計數所有vocabulary出現的次數
        )

    intent2idx = {tag: i for i, tag in enumerate(intents)} #紀錄一個dict,i為value，intents內容(tag)為dict的key
    intent_tag_path = args.output_dir / "intent2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2))#將intent2idx記錄成json儲存
    logging.info(f"Intent 2 index saved at {str(intent_tag_path.resolve())}")

    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
