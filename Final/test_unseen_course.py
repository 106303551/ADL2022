# test
import pandas as pd
from deepctr.layers import custom_objects
from tensorflow.python.keras.models import  save_model,load_model
import pandas as pd

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

unseen_test = pd.read_csv('./data/course/mmoe_test_unseen_course.csv').drop(columns=['user_id','idx','course_id'])
feature_names =unseen_test.columns.tolist()
new_feature_list=[]
for i in range(len(feature_names)):
    new_feature = str(i)+"_feature"
    new_feature_list.append(new_feature)
feature_names = new_feature_list
unseen_test.columns = feature_names


test_unseen_model_input = {name: unseen_test[name] for name in feature_names}

label_name=[i for i in range(728)]

model = load_model('./ckpt/unseen_course/PLE.h5',custom_objects)# load_model,just add a parameter
pred_ans = model.predict(test_unseen_model_input, batch_size=128)
unseen_test = pd.read_csv('./data/course/mmoe_test_unseen_course.csv',usecols=['user_id'])
for idx,label in enumerate(label_name):
    unseen_test[label] = pred_ans[idx]
unseen_test.to_csv('./data/course/mmoe_pred_test_unseen_course.csv',index=False)