import pandas as pd
def mmoe_train_label_multiclass():
    data = pd.read_csv('./data/course/mmoe_train_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_train_course_label_multi_class.csv',index=False)

def mmoe_train_label_multilabel():
    data = pd.read_csv('./data/course/mmoe_train_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_train_course_label_multi_label.csv',index=False)

def mmoe_val_seen_label_multiclass():
    data = pd.read_csv('./data/course/mmoe_val_seen_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_val_seen_course_label_multi_class.csv',index=False)

def mmoe_val_seen_label_multilabel():
    data = pd.read_csv('./data/course/mmoe_val_seen_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_val_seen_course_label_multi_label.csv',index=False)

def mmoe_val_unseen_label_multiclass():
    data = pd.read_csv('./data/course/mmoe_val_unseen_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_val_unseen_course_label_multi_class.csv',index=False)

def mmoe_val_unseen_label_multilabel():
    data = pd.read_csv('./data/course/mmoe_val_unseen_course.csv',usecols=['course_id']).dropna()
    course_info = pd.read_csv("./data/course_list.csv")
    label_list = course_info['idx'].tolist()
    label = data['course_id']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx))] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/course/mmoe_val_unseen_course_label_multi_label.csv',index=False)

mmoe_train_label_multiclass()
mmoe_train_label_multilabel()
mmoe_val_seen_label_multiclass()
mmoe_val_seen_label_multilabel()
mmoe_val_unseen_label_multiclass()
mmoe_val_unseen_label_multilabel()
