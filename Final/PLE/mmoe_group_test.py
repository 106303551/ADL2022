#test
import pandas as pd
from tensorflow.python.keras.models import  save_model,load_model
from deepctr.layers import custom_objects
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


seen_test = pd.read_csv('./drive/MyDrive/data/subgroup/mmoe_test_seen_subgroup.csv').drop(columns=['user_id','idx'])
feature_names =seen_test.columns.tolist()
new_feature_list=[]
for i in range(len(feature_names)):
    new_feature = str(i)+"_feature"
    new_feature_list.append(new_feature)
feature_names = new_feature_list

seen_test.columns = feature_names
unseen_test = pd.read_csv('./drive/MyDrive/data/subgroup/mmoe_test_unseen_subgroup.csv').drop(columns=['user_id','idx'])
unseen_test.columns = feature_names


test_seen_model_input = {name: seen_test[name] for name in feature_names}
test_unseen_model_input = {name: unseen_test[name] for name in feature_names}

label_name=[i for i in range(728)]


model = load_model('./drive/MyDrive/data/PLE.h5',custom_objects)# load_model,just add a parameter
pred_ans = model.predict(test_seen_model_input, batch_size=256)
seen_test = pd.read_csv('./drive/MyDrive/data/subgroup/mmoe_test_seen_subgroup.csv',usecols=['user_id'])
for idx,label in enumerate(label_name):
    seen_test[label] = pred_ans[idx]
seen_test.to_csv('./drive/MyDrive/data/subgroup/mmoe_pred_test_seen_subgroup.csv',index=False)

pred_ans = model.predict(test_unseen_model_input, batch_size=256)
unseen_test = pd.read_csv('./drive/MyDrive/data/subgroup/mmoe_test_unseen_subgroup.csv',usecols=['user_id'])
for idx,label in enumerate(label_name):
    unseen_test[label] = pred_ans[idx]
unseen_test.to_csv('./drive/MyDrive/data/subgroup/mmoe_pred_test_unseen_subgroup.csv',index=False)