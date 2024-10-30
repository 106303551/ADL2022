import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

RECORDS = ['train', 'val_seen', 'val_unseen']
SPLITS = ['test_seen']

class HahowDataset(Dataset):
    def __init__(
        self,
        data,
        users,
        courses,
        label_mapping,
        bought_courses,
    ):
        self.data = data
        self.users = users
        self.courses = courses
        self.label_mapping = label_mapping
        self.bought_courses = bought_courses
        self.idx2label = {idx: course for course, idx in self.label_mapping.items()}
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
        if self.bought_courses[uid] != []:
            text += '。購買課程：' + ','.join([self.courses[self.courses['course_id'] == self.idx2label[cid]]['course_name'].values[0] for cid in self.bought_courses[uid]])

        return {
            'text': text,
            'id': uid,
        }

def main(args):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(f'Using {args.device} for test.')

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
        split: HahowDataset(split_data, df_users, df_courses, course2idx, bought_courses) 
        for split, split_data in data.items()
    }
    dataloaders = {
        split: DataLoader(dataset=split_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for split, split_dataset in datasets.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(course2idx)).to(args.device)

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            print(f'Processing {split}.csv ...')

            prediction = []
            for batch in tqdm(dataloaders[split]):
                text = batch['text']
                text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).to(args.device)
                pred = model(**text)
                
                for id, p in zip(batch['id'], pred.logits.detach().cpu().numpy()):
                    for cidx in bought_courses[id]:
                        p[cidx] = float('-inf')
                    p = np.argsort(p)[::-1].tolist()
                    prediction.append(f'{id},{" ".join([datasets[split].idx2label[v] for v in p[:50]])}')

            with open(args.pred_file, 'w') as f:
                f.write('user_id,course_id\n')
                for pred in prediction:
                    f.write(f'{pred}\n')

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
        default='./ckpt/seen_course/'
    )
    parser.add_argument('--pred_file', type=Path, default='pred_seen_course.csv')

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
