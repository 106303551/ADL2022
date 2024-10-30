import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

RECORDS = ['train_group']
SPLITS = ['train_group', 'val_seen_group', 'val_unseen_group']

class HahowDataset(Dataset):
    def __init__(
        self,
        data,
        split,
        users,
        subgroups,
        bought_subgroups,
    ):
        self.data = data
        self.split = split
        self.users = users
        self.subgroups = subgroups
        self.bought_subgroups = bought_subgroups
        self.column_mapping = {'gender': '性別：', 'occupation_titles': '職業：', 'interests': '興趣：', 'recreation_names': '喜好：'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid, gids = self.data.iloc[index]['user_id'], str(self.data.iloc[index]['subgroup']).split(' ')
        text = self.users[self.users['user_id']==uid].drop(columns=['user_id']).replace({'male': '男', 'female': '女', 'other': np.nan}).dropna(axis=1)
        for k, v in self.column_mapping.items():
            if k in text.columns:
                text[k] = v + text[k].values[0]
        text = '。'.join(text.values.tolist()[0])
        if self.split == SPLITS[1] and self.bought_subgroups[uid] != []:
            text += '。購買課程子分類：' + ','.join([self.subgroups[self.subgroups['subgroup_id'] == gid + 1]['subgroup_name'].values[0] for gid in self.bought_subgroups[uid]])
        label = torch.zeros(len(self.subgroups))
        for gid in gids:
            label[int(gid) - 1] = 1 / len(gids)

        return {
            'text': text,
            'label': label,
            'id': uid,
        }

def apk(actual, predicted, k=50):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def main(args):
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
        torch.cuda.manual_seed_all(args.rand_seed)  
    np.random.seed(args.rand_seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(f'Using {args.device} for training.')

    df_subgroups = pd.read_csv(args.data_dir / 'subgroups.csv')

    data = {record: pd.read_csv(args.data_dir / f'{record}.csv').dropna() for record in RECORDS}
    bought_subgroups = defaultdict(list)
    for record_data in data.values():
        for _, row in record_data.iterrows():
            bought_subgroups[row['user_id']] += [int(gid) - 1 for gid in str(row['subgroup']).split(' ')]
    
    df_users = pd.read_csv(args.data_dir / 'users.csv')
    data = {split: pd.read_csv(args.data_dir / f'{split}.csv').dropna() for split in SPLITS}
    datasets = {
        split: HahowDataset(split_data, split, df_users, df_subgroups, bought_subgroups) 
        for split, split_data in data.items()
    }
    dataloaders = {
        split: DataLoader(dataset=split_dataset, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=args.num_workers)
        for split, split_dataset in datasets.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=len(df_subgroups)).to(args.device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(dataloaders[SPLITS[0]])*args.num_epoch)
    
    for epoch in range(args.num_epoch):
        print('==================== Epoch {:1d}/{} ===================='.format(epoch + 1, args.num_epoch))
        start = time.time()

        total_loss, total_ap = 0, 0
        model.train()
        for i, batch in enumerate(dataloaders[SPLITS[0]]):
            text, label = batch['text'], batch['label'].to(args.device)
            text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
            pred = model(**text, labels=label)
            loss = criterion(pred.logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()

            total_loss += loss.detach().item()
            for a, p in zip(label, pred.logits):
                total_ap += apk(np.nonzero(a.cpu().numpy())[0].tolist(), np.argsort(p.detach().cpu().numpy())[::-1].tolist())

            if (i + 1) % 100 == 0:
                print('Step {:4d}/{} Training Loss/MAP {:.6f}/{:.6f}'.format(i + 1, len(dataloaders[SPLITS[0]]), total_loss / (i + 1), total_ap / (i + 1) / len(label)))
        loss_t, map_t = total_loss / len(dataloaders[SPLITS[0]]), total_ap / len(dataloaders[SPLITS[0]].dataset)
        print('Training Loss/MAP {:.6f}/{:.6f} Epoch Time {:.6f}s'.format(loss_t, map_t, time.time() - start))

        if epoch != args.num_epoch - 1:
            model.save_pretrained(args.ckpt_dir_seen)
        else:
            model.save_pretrained(args.ckpt_dir_unseen)

    model.eval()
    with torch.no_grad():
        subgroups_embedding = torch.zeros((len(df_subgroups), 768))
        for i, row in df_subgroups.iterrows():
            text = row['subgroup_name']
            text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
            text = model(**text, output_hidden_states=True)
            subgroups_embedding[row['subgroup_id'] - 1] = torch.mean(text.hidden_states[-1], dim=1)
        similarity_matrix = cosine_similarity(subgroups_embedding)
        
        for split in SPLITS[1:]:
            total_loss, total_ap, total_ap2, total_ap3 = 0, 0, 0, 0
            _total_ap = [0]*100
            for batch in tqdm(dataloaders[split]):
                text, label = batch['text'], batch['label'].to(args.device)
                text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
                pred = model(**text, labels=label)
                loss = criterion(pred.logits, label)

                total_loss += loss.detach().item()
                for id, a, p in zip(batch['id'], label, pred.logits):
                    classification = np.argsort(p.detach().cpu().numpy())[::-1].tolist()
                    total_ap += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)

                    if split == SPLITS[1]:
                        if bought_subgroups[id] != []:
                            similarity = np.argsort(np.max(similarity_matrix[bought_subgroups[id]], axis=0))[::-1].tolist()
                            total_ap2 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), similarity)
                            
                            curr_rank = 1
                            new_rank = [0] * len(similarity)
                            for s, c in zip(similarity, classification):
                                new_rank[s] += curr_rank / 16
                                new_rank[c] += curr_rank
                                curr_rank += 1
                            total_ap3 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), np.argsort(new_rank))

                            for i in range(100):
                                curr_rank = 1
                                new_rank = [0] * len(similarity)
                                for s, c in zip(similarity, classification):
                                    new_rank[s] += curr_rank / (i + 1)
                                    new_rank[c] += curr_rank
                                    curr_rank += 1
                                _total_ap[i] += apk(np.nonzero(a.cpu().numpy())[0].tolist(), np.argsort(new_rank))
                        else:
                            total_ap2 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)
                            total_ap3 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)

                            for i in range(100):
                                _total_ap[i] += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)
            print(_total_ap)
            loss_e, map_e, map_e2, map_e3 = total_loss / len(dataloaders[split]), total_ap / len(dataloaders[split].dataset), total_ap2 / len(dataloaders[split].dataset), total_ap3 / len(dataloaders[split].dataset)
            print('Validation ({}) Loss/MAP_C/MAP_S/MAP {:.6f}/{:.6f}/{:.6f}/{:.6f}'.format(split.split('_')[1].capitalize(), loss_e, map_e, map_e2, map_e3))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory to the dataset.',
        default='./data/',
    )
    parser.add_argument(
        '--ckpt_dir_seen',
        type=Path,
        help='Directory to save the model file.',
        default='./ckpt/seen_group/',
    )
    parser.add_argument(
        '--ckpt_dir_unseen',
        type=Path,
        help='Directory to save the model file.',
        default='./ckpt/unseen_group/',
    )

    # data
    parser.add_argument('--max_len', type=int, default=512)

    # model
    parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese')

    # optimizer
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # data loader
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)

    # training
    parser.add_argument(
        '--device', type=torch.device, help='cpu, cuda, cuda:0, cuda:1', default='cuda:0'
    )
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, help='Random seed.', default=13)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir_seen.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir_unseen.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args)
