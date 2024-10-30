import pandas as pd

def gen_mmoe_train():
    course_info = pd.read_csv('./data/course_list.csv')
    course_info.set_index('course_id',inplace=True)
    df = pd.read_csv('./data/users/mmoe_users.csv')
    df_id = df['user_id'].values.tolist()
    train = pd.read_csv('./data/train.csv').dropna()
    train_id = train['user_id'].values.tolist()
    train.set_index('user_id',inplace=True)
    course_id =['0' for i in range(df.shape[0])]
    df['course_id'] = course_id

    drop_list=[]
    for idx,id in enumerate(df_id):
        if id not in train_id:
            drop_list.append(idx)
            continue
        course = train.loc[id]['course_id']
        course = course.split(' ')
        course_idx=""
        for v in course:
            c_idx = course_info.loc[v]['idx']
            course_idx+=str(c_idx)+" "
        df['course_id'][idx] = course_idx
    seen_df = df.drop(index=drop_list)
    seen_df.to_csv('./data/course/mmoe_train_course.csv',index=False)

def gen_mmoe_val():
    course_info = pd.read_csv('./data/course_list.csv')
    course_info.set_index('course_id',inplace=True)
    df = pd.read_csv('./data/users/mmoe_users.csv')
    df_id = df['user_id'].values.tolist()
    val_seen = pd.read_csv('./data/val_seen.csv').dropna()
    val_unseen = pd.read_csv('./data/val_unseen.csv').dropna()
    val_seen_id = val_seen['user_id'].values.tolist()
    val_unseen_id = val_unseen['user_id'].values.tolist()
    val_seen.set_index('user_id',inplace=True)
    val_unseen.set_index('user_id',inplace=True)
    course_id =['0' for i in range(df.shape[0])]
    df['course_id'] = course_id

    drop_list=[]
    for idx,id in enumerate(df_id):
        if id not in val_seen_id:
            drop_list.append(idx)
            continue
        course = val_seen.loc[id]['course_id']
        course = course.split(' ')
        course_idx=""
        for v in course:
            c_idx = course_info.loc[v]['idx']
            course_idx+=str(c_idx)+" "
        df['course_id'][idx] = course_idx
    seen_df = df.drop(index=drop_list)
    seen_df.to_csv('./data/course/mmoe_val_seen_course.csv',index=False)

    drop_list=[]
    for idx,id in enumerate(df_id):
        if id not in val_unseen_id:
            drop_list.append(idx)
            continue
        course = val_unseen.loc[id]['course_id']
        course = course.split(' ')
        course_idx=""
        for v in course:
            c_idx = course_info.loc[v]['idx']
            course_idx+=str(c_idx)+" "
        df['course_id'][idx] = course_idx
    seen_df = df.drop(index=drop_list)
    seen_df.to_csv('./data/course/mmoe_val_unseen_course.csv',index=False)

def gen_mmoe_test():
    course_info = pd.read_csv('./data/course_list.csv')
    course_info.set_index('course_id',inplace=True)
    df = pd.read_csv('./data/users/mmoe_users.csv')
    df_id = df['user_id'].values.tolist()
    val_seen = pd.read_csv('./data/test_seen.csv').dropna()
    val_unseen = pd.read_csv('./data/test_unseen.csv').dropna()
    val_seen_id = val_seen['user_id'].values.tolist()
    val_unseen_id = val_unseen['user_id'].values.tolist()
    val_seen.set_index('user_id',inplace=True)
    val_unseen.set_index('user_id',inplace=True)
    course_id =['0' for i in range(df.shape[0])]
    df['course_id'] = course_id

    drop_list=[]
    for idx,id in enumerate(df_id):
        if id not in val_seen_id:
            drop_list.append(idx)
            continue
    seen_df = df.drop(index=drop_list)
    seen_df.to_csv('./data/course/mmoe_test_seen_course.csv',index=False)

    drop_list=[]
    for idx,id in enumerate(df_id):
        if id not in val_unseen_id:
            drop_list.append(idx)
            continue
    unseen_df = df.drop(index=drop_list)
    unseen_df.to_csv('./data/course/mmoe_test_unseen_course.csv',index=False)

gen_mmoe_train()
gen_mmoe_val()
gen_mmoe_test()

