import pandas as pd
def gen_mmoe_train_label():
    data = pd.read_csv('./data/subgroup/mmoe_train_subgroup.csv',usecols=['subgroups']).dropna()
    subgroup_info = pd.read_csv("./data/subgroup/subgroups.csv")
    label_list = subgroup_info['subgroup_id'].tolist()
    label = data['subgroups']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_train_subgroup_label_multi_class.csv',index=False)

    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_train_subgroup_label_multi_label.csv',index=False)

def gen_mmoe_val_seen_label():
    data = pd.read_csv('./data/subgroup/mmoe_val_seen_subgroup.csv',usecols=['subgroups']).dropna()
    subgroup_info = pd.read_csv("./data/subgroup/subgroups.csv")
    label_list = subgroup_info['subgroup_id'].tolist()
    label = data['subgroups']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_val_seen_subgroup_label_multi_class.csv',index=False)

    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_val_seen_subgroup_label_multi_label.csv',index=False)

def gen_mmoe_val_unseen_label():
    data = pd.read_csv('./data/subgroup/mmoe_val_unseen_subgroup.csv',usecols=['subgroups']).dropna()
    subgroup_info = pd.read_csv("./data/subgroup/subgroups.csv")
    label_list = subgroup_info['subgroup_id'].tolist()
    label = data['subgroups']
    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1/len(l)
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_val_unseen_subgroup_label_multi_class.csv',index=False)

    new_l_list = []
    for i,l in enumerate(label):
        new_l = [0]*len(label_list)
        l = l.rstrip()
        if pd.isna(l):
            continue
        l = l.split(' ')
        for idx in l:
            new_l[(int(idx)-1)] = 1
        new_l_list.append(new_l)
    new_label = pd.DataFrame(new_l_list,columns=label_list)
    new_label.to_csv('./data/subgroup/mmoe_val_unseen_subgroup_label_multi_label.csv',index=False)

gen_mmoe_train_label()
gen_mmoe_val_seen_label()
gen_mmoe_val_unseen_label()