import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict
import re

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

RECORDS = ['train']
SPLITS = ['train', 'val_seen', 'val_unseen']

class HahowDataset(Dataset):
    def __init__(
        self,
        data,
        split,
        users,
        courses,
        label_mapping,
        bought_courses,
    ):
        self.data = data
        self.split = split
        self.users = users
        self.courses = courses
        self.label_mapping = label_mapping
        self.bought_courses = bought_courses
        self.idx2label = {idx: course for course, idx in self.label_mapping.items()}
        self.column_mapping = {'gender': '性別：', 'occupation_titles': '職業：', 'interests': '興趣：', 'recreation_names': '喜好：'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid, cids = self.data.iloc[index]['user_id'], self.data.iloc[index]['course_id'].split(' ')
        text = self.users[self.users['user_id']==uid].drop(columns=['user_id']).replace({'male': '男', 'female': '女', 'other': np.nan}).dropna(axis=1)
        for k, v in self.column_mapping.items():
            if k in text.columns:
                text[k] = v + text[k].values[0]
        text = '。'.join(text.values.tolist()[0])
        if self.split == SPLITS[0]:
            bids = cids[:len(cids) // 2]
            if bids != []:
                text += '。購買課程：' + ','.join([self.courses[self.courses['course_id'] == bid]['course_name'].values[0] for bid in bids])
        elif self.split == SPLITS[1] and self.bought_courses[uid] != []:
            text += '。購買課程：' + ','.join([self.courses[self.courses['course_id'] == self.idx2label[cid]]['course_name'].values[0] for cid in self.bought_courses[uid]])
        label = torch.zeros(len(self.label_mapping))
        for cid in cids:
            label[self.label_mapping[cid]] = 1 / len(cids)

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

    df_courses = pd.read_csv(args.data_dir / 'courses.csv')
    courses = set(df_courses['course_id'])
    course2idx = {course: i for i, course in enumerate(sorted(courses))}

    data = {record: pd.read_csv(args.data_dir / f'{record}.csv') for record in RECORDS}
    bought_courses = defaultdict(list)
    for record_data in data.values():
        for _, row in record_data.iterrows():
            bought_courses[row['user_id']] += [course2idx[cid] for cid in row['course_id'].split(' ')]
    
    df_users = pd.read_csv(args.data_dir / 'users.csv')
    data = {split: pd.read_csv(args.data_dir / f'{split}.csv') for split in SPLITS}
    datasets = {
        split: HahowDataset(split_data, split, df_users, df_courses, course2idx, bought_courses) 
        for split, split_data in data.items()
    }
    dataloaders = {
        split: DataLoader(dataset=split_dataset, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=args.num_workers)
        for split, split_dataset in datasets.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=len(course2idx)).to(args.device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(dataloaders[SPLITS[0]])*(args.num_epoch+2))
    
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

        model.save_pretrained(args.ckpt_dir)

    model.eval()
    with torch.no_grad():
        column_mapping = {'course_name': '名稱：', 'groups': '分類：', 'sub_groups': '子分類：', 'topics': '主題：', 'description': '詳情：'}
        courses_embedding = torch.zeros((len(course2idx), 768))
        for i, row in df_courses.iterrows():
            text = row[column_mapping.keys()].dropna()
            for k, v in column_mapping.items():
                if k in text.index:
                    text[k] = v + re.compile(r'<.*?>').sub('', text[k])
            text = '。'.join(text.values.tolist())
            text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
            text = model(**text, output_hidden_states=True)
            courses_embedding[course2idx[row['course_id']]] = torch.mean(text.hidden_states[-1], dim=1)
        similarity_matrix = cosine_similarity(courses_embedding)

        for split in SPLITS[1:]:
            total_loss, total_ap, total_ap2, total_ap3 = 0, 0, 0, 0
            for batch in tqdm(dataloaders[split]):
                text, label = batch['text'], batch['label'].to(args.device)
                text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
                pred = model(**text, labels=label)
                loss = criterion(pred.logits, label)

                total_loss += loss.detach().item()
                for id, a, p in zip(batch['id'], label, pred.logits):
                    if split == SPLITS[0]:
                        for cidx in bought_courses[id]:
                            p[cidx] = float('-inf')
                    classification = np.argsort(p.detach().cpu().numpy())[::-1].tolist()
                    total_ap += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)

                    if split == SPLITS[1]:
                        if bought_courses[id] != []:
                            similarity = np.max(similarity_matrix[bought_courses[id]], axis=0)
                            similarity[bought_courses[id]] = float('-inf')
                            similarity = np.argsort(similarity)[::-1].tolist()
                            total_ap2 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), similarity)
                            
                            curr_rank = 1
                            new_rank = [0] * len(similarity)
                            for s, c in zip(similarity, classification):
                                new_rank[s] += curr_rank / 64
                                new_rank[c] += curr_rank
                                curr_rank += 1
                            total_ap3 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), np.argsort(new_rank))
                        else:
                            total_ap2 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)
                            total_ap3 += apk(np.nonzero(a.cpu().numpy())[0].tolist(), classification)
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
        '--ckpt_dir',
        type=Path,
        help='Directory to save the model file.',
        default='./ckpt/seen_course/',
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
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--rand_seed', type=int, help='Random seed.', default=13)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main(args)
