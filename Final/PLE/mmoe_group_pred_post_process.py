import pandas as pd
import numpy as np
df = pd.read_csv('./data/mmoe_pred_test_unseen_subgroup.csv')
pred_text_list=[]

for idx,row in df.iterrows():
    pred_text = row['user_id']+","
    value = row.values[1:]
    label_idx = [int(v+1) for v in value.argsort()[::-1][:50]]
    for l in label_idx:
        pred_text+=str(l)+" "
    pred_text = pred_text.rstrip()
    pred_text_list.append(pred_text)

with open(f'pred_unseen_group.csv', 'w') as f:
    f.write('user_id,subgroup\n')
    for pred in pred_text_list:
        f.write(f'{pred}\n')

    