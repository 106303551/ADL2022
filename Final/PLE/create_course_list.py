import pandas as pd 

df = pd.read_csv('./data/courses.csv',usecols=['course_id'])
df['idx'] = [i for i in range(df.shape[0])]
df.to_csv('./data/course_list.csv ',index=False)
