import pandas as pd
import numpy as np
df = pd.read_csv('./data/course/mmoe_pred_test_unseen_course.csv')
course_list = pd.read_csv('./data/course_list.csv')
course_list.set_index('idx')
course_list_2 = pd.read_csv('./data/course_list.csv')
course_list_2.set_index('idx',inplace=True)
pred_text_list=[]

for idx,row in df.iterrows():
    pred_text = row['user_id']+","
    value = row.values[1:]
        
    label_idx = [int(v) for v in value.argsort()[::-1][:50]]
    for l in label_idx:
        pred_text+=str(course_list.loc[int(l)]['course_id'])+" "
    pred_text = pred_text.rstrip()
    pred_text_list.append(pred_text)

with open(f'pred_unseen_course.csv', 'w') as f:
    f.write('user_id,course_id\n')
    for pred in pred_text_list:
        f.write(f'{pred}\n')