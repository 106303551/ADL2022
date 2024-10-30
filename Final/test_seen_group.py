import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

RECORDS = ['train_group', 'val_seen_group', 'val_unseen_group']
SPLITS = ['test_seen_group']

class HahowDataset(Dataset):
    def __init__(
        self,
        data,
        users,
        subgroups,
        bought_subgroups,
    ):
        self.data = data
        self.users = users
        self.subgroups = subgroups
        self.bought_subgroups = bought_subgroups
        self.column_mapping = {'gender': '性別：', 'occupation_titles': '職業：', 'interests': '興趣：', 'recreation_names': '喜好：'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid = self.data.iloc[index]['user_id']
        text = self.users[self.users['user_id']==uid].drop(columns=['user_id']).replace({'male': '男', 'female': '女', 'other': np.nan}).dropna(axis=1)
        for k, v in self.column_mapping.items():
            if k in text.columns:
                text[k] = v + text[k].values[0]
        text = '。'.join(text.values.tolist()[0])
        if self.bought_subgroups[uid] != []:
            text += '。購買課程子分類：' + ','.join([self.subgroups[self.subgroups['subgroup_id'] == gid + 1]['subgroup_name'].values[0] for gid in self.bought_subgroups[uid]])

        return {
            'text': text,
            'id': uid,
        }

def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(f'Using {args.device} for test.')

    df_subgroups = pd.read_csv(args.data_dir / 'subgroups.csv')

    data = {record: pd.read_csv(args.data_dir / f'{record}.csv').dropna() for record in RECORDS}
    bought_subgroups = defaultdict(list)
    for record_data in data.values():
        for _, row in record_data.iterrows():
            bought_subgroups[row['user_id']] += [int(gid) - 1 for gid in str(row['subgroup']).split(' ')]
    
    df_users = pd.read_csv(args.data_dir / 'users.csv')
    data = {split: pd.read_csv(args.data_dir / f'{split}.csv') for split in SPLITS}
    datasets = {
        split: HahowDataset(split_data, df_users, df_subgroups, bought_subgroups)
        for split, split_data in data.items()
    }
    dataloaders = {
        split: DataLoader(dataset=split_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for split, split_dataset in datasets.items()
    }

    cs = torch.nn.CosineSimilarity(dim=0)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(df_subgroups)).to(args.device)

    model.eval()
    with torch.no_grad():
        subgroups_embedding = torch.zeros((len(df_subgroups), 768))
        for i, row in df_subgroups.iterrows():
            text = row['subgroup_name']
            text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
            text = model(**text, output_hidden_states=True)
            subgroups_embedding[row['subgroup_id'] - 1] = torch.mean(text.hidden_states[-1], dim=1)
        similarity_matrix = cosine_similarity(subgroups_embedding)

        for split in SPLITS:
            print(f'Processing {split}.csv ...')

            prediction = {}
            for batch in tqdm(dataloaders[split]):
                text = batch['text']
                text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
                pred = np.argsort(model(**text).logits.detach().cpu().numpy())[:, ::-1].tolist()

                for id, p in zip(batch['id'], pred):
                    prediction[id] = p

            if split == SPLITS[0]:
                for _, row in tqdm(data[split].iterrows()):
                    uid = row['user_id']
                    if bought_subgroups[uid] != []:
                        similarity = np.argsort(np.max(similarity_matrix[bought_subgroups[uid]], axis=0))[::-1].tolist()
                        classification = prediction[uid]

                        curr_rank = 1
                        new_rank = [0] * len(similarity)
                        for a, b in zip(similarity, classification):
                            new_rank[a] += curr_rank / 32
                            new_rank[b] += curr_rank
                            curr_rank += 1
                        prediction[uid] = np.argsort(new_rank)

            with open(args.pred_file, 'w') as f:
                f.write('user_id,subgroup\n')
                for id, pred in prediction.items():
                    f.write(f'{id},{" ".join([str(v + 1) for v in pred[:50]])}\n')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory to the dataset.',
        default='./data/',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=Path,
        help='Path to model checkpoint.',
        default='./ckpt/seen_group/'
    )
    parser.add_argument('--pred_file', type=Path, default='pred_seen_group.csv')

    # data
    parser.add_argument('--max_len', type=int, default=512)

    # model
    parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese')

    # data loader
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)

    # test
    parser.add_argument(
        '--device', type=torch.device, help='cpu, cuda, cuda:0, cuda:1', default='cuda:0'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args)
