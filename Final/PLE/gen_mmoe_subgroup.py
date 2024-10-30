import pandas as pd
def gen_mmoe_train():
    user = pd.read_csv('./data/users/mmoe_users.csv')
    user.set_index('idx',inplace=True)
    train_group = pd.read_csv('./data/train_group.csv').dropna()
    train_group['idx'] = [i for i in range(train_group.shape[0])]
    train_id = train_group['user_id'].values.tolist()
    train_group.set_index('user_id',inplace=True)
    drop_list=[]
    subgroup_list=[]
    for idx,row in user.iterrows():
        user_id = row['user_id']
        if user_id not in train_id:
            drop_list.append(idx)
            continue
        subgroup_list.append(train_group.loc[user_id]['subgroup'])
    user = user.drop(index=drop_list)
    user['subgroups'] = subgroup_list
    user.to_csv('./data/subgroup/mmoe_train_subgroup.csv',index=False)

def gen_mmoe_val_seen():
    user = pd.read_csv('./data/users/mmoe_users.csv')
    user.set_index('idx',inplace=True)
    val_seen_group = pd.read_csv('./data/val_seen_group.csv').dropna()
    val_seen_group['idx'] = [i for i in range(val_seen_group.shape[0])]
    val_seen_id = val_seen_group['user_id'].values.tolist()
    val_seen_group.set_index('user_id',inplace=True)
    drop_list=[]
    subgroup_list=[]
    for idx,row in user.iterrows():
        user_id = row['user_id']
        if user_id not in val_seen_id:
            drop_list.append(idx)
            continue
        subgroup_list.append(val_seen_group.loc[user_id]['subgroup'])
    user = user.drop(index=drop_list)
    user['subgroups'] = subgroup_list
    user.to_csv('./data/subgroup/mmoe_val_seen_subgroup.csv',index=False)

def gen_mmoe_val_unseen():
    user = pd.read_csv('./data/users/mmoe_users.csv')
    user.set_index('idx',inplace=True)
    val_unseen_group = pd.read_csv('./data/val_unseen_group.csv').dropna()
    val_unseen_group['idx'] = [i for i in range(val_unseen_group.shape[0])]
    val_unseen_id = val_unseen_group['user_id'].values.tolist()
    val_unseen_group.set_index('user_id',inplace=True)
    drop_list=[]
    subgroup_list=[]
    for idx,row in user.iterrows():
        user_id = row['user_id']
        if user_id not in val_unseen_id:
            drop_list.append(idx)
            continue
        subgroup_list.append(val_unseen_group.loc[user_id]['subgroup'])
    user = user.drop(index=drop_list)
    user['subgroups'] = subgroup_list
    user.to_csv('./data/subgroup/mmoe_val_unseen_subgroup.csv',index=False)

def gen_mmoe_test():
    user = pd.read_csv('./data/users/mmoe_users.csv')
    user.set_index('idx',inplace=True)
    seen_test_group = pd.read_csv('./data/test_seen_group.csv').dropna()
    unseen_test_group = pd.read_csv('./data/test_unseen_group.csv').dropna()
    seen_test_group['idx'] = [i for i in range(seen_test_group.shape[0])]
    seen_id = seen_test_group['user_id'].values.tolist()
    seen_test_group.set_index('user_id',inplace=True)
    unseen_test_group['idx'] = [i for i in range(unseen_test_group.shape[0])]
    unseen_id = unseen_test_group['user_id'].values.tolist()
    unseen_test_group.set_index('user_id',inplace=True)

    drop_list=[]
    for idx,row in user.iterrows():
        user_id = row['user_id']
        if user_id not in seen_id:
            drop_list.append(idx)
            continue
    seen_user = user.drop(index=drop_list)
    seen_user.to_csv('./data/subgroup/mmoe_test_seen_subgroup.csv',index=False)

    drop_list=[]
    for idx,row in user.iterrows():
        user_id = row['user_id']
        if user_id not in unseen_id:
            drop_list.append(idx)
            continue
    unseen_user = user.drop(index=drop_list)
    unseen_user.to_csv('./data/subgroup/mmoe_test_unseen_subgroup.csv',index=False)

gen_mmoe_train()
gen_mmoe_val_seen()
gen_mmoe_val_unseen()
gen_mmoe_test()

    


