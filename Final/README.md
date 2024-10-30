# Final Project ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-final"
make
conda activate adl-final
pip install -r requirements.txt

# Otherwise
pip install -r requirements.in
```
## BERT
### Reproducing
```shell
# Please put all csv files into ./data/ folder
bash download.sh
bash run.sh
# After that, four prediction files will be in the working directory, including pred_seen_course.csv, pred_unseen_course.csv, pred_seen_group.csv, pred_unseen_group.csv
```

### Training
```shell
# Seen Course
python train_seen_course.py --data_dir /path/to/dataset_directory

# Seen Topic & Unseen Topic (Simultaneously)
python train_group.py --data_dir /path/to/dataset_directory
```

### Test
```shell
# Seen Course
python test_seen_course.py --data_dir /path/to/dataset_directory --pred_file /path/to/pred.csv

# Unseen Course
python test_unseen_course.py
python3.9 mmoe_course_pred_post_process.py

# Seen Topic
python test_seen_group.py --data_dir /path/to/dataset_directory --pred_file /path/to/pred.csv

# Unseen Topic
python test_unseen_group.py --data_dir /path/to/dataset_directory --pred_file /path/to/pred.csv
```
## PLE

### data preprocess
```shell
bash data_preprocess.sh #要跑很久

```
### Training(By Colab)
```shell
1.上傳mmoe_course.ipynb及mmoe_group.ipynb至colab
2.自定義資料路徑
3.執行mmoe_course及mmoe_group進行訓練
4.mmoe_course及mmoe_grooup裡也有Test部分
```
### Testing
```shell
1.自定義資料路徑及模型路徑
2.執行mmoe_course_test及mmoe_group_test進行Predict
```

### Data postprocess
```shell
1.自定義資料路徑
2.執行mmoe_course_pred_post_process 及mmoe_group_pred_post_process
```
