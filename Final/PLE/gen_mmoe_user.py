import pandas as pd

user = pd.read_csv('./data/users/users.csv')
user_col = user.columns.values.tolist()
user_col.remove('user_id')
gender2idx = {'male':0,'female':1,'other':2}
recreation = pd.read_csv('./data/recreations.csv')
occupation_titles = pd.read_csv('./data/occupation_titles.csv')
interests = pd.read_csv('./data/interests.csv')
new_col = ['gender']+interests['interest_name'].values.tolist()+recreation['name'].values.tolist()+occupation_titles['name'].values.tolist()
new_user = pd.DataFrame(columns=new_col)

for i,row in user.iterrows():
    new_row = [0]*len(new_col)
    for col in user_col:
        if col == 'gender':
            if pd.isna(row[col]) == True:
                row[col]='other'
            new_row[0] = gender2idx[row[col]]
            continue
        if pd.isna(row[col]) == False:
            v_list = row[col].split(',')
            for v in v_list:
                idx = new_col.index(v)
                new_row[idx] = 1
    new_user.loc[i] = new_row

user_id = user['user_id'].values.tolist()
new_user['user_id'] = user_id
new_user['idx'] = [i for i in range(len(user_id))]
new_col = ['idx']+['user_id']+new_col
new_user = new_user[new_col]
new_user.to_csv('./data/users/mmoe_users.csv',index=False)
