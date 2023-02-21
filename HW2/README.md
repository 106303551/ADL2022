# HW 2 

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw2"
make
conda activate adl-hw2
pip install -r requirements.txt
```



## Context Selection

### Train
```shell
python MC_train.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
```
* model_name_or_path: Directory to model or identifier for huggingface.co/models.EX:hfl/chinese-roberta-wwm-ext
* output_dir: Directory to checkpoints and trained model.
* train_file:Directory to train file .
* validation_file: Directory to validation file.
*context_file: Directory to context file.
*cache_dir:Directory to cache.
*max_seq_length:the maximum length for feature.
*num_train_epochs:numer of train epochs.
*warmup_ratio:warmup_steps=warmup_ratio*total_steps

### Test

```shell
python MC_test.py \
  --model_name_or_path <model_name_or_path> \
  --test_file <test_file> \
  --context_file <context_file> \
  --output_file <output_file> \
  --cache_dir ./cache/ \
  --max_seq_length 512 \ 
```

* model_name_or_path: Directory to model or identifier for huggingface.co/models.EX:hfl/chinese-roberta-wwm-ext
* output_dir: Directory to checkpoints and trained model.
* test_file:Directory to test file .
*context_file: Directory to context file.
*output_file: Directory to output file.
*cache_dir:Directory to cache.
*max_seq_length:the maximum length for feature.
*num_train_epochs:numer of train epochs.
*warmup_ratio:warmup_steps=warmup_ratio*total_steps


## Question Answering

### Train
```shell
python QA_train.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --cache_dir ./cache/ \
  --per_device_train_batch_size 4\
  --per_device_eval_batch_size 4\
  --gradient_accumulation_steps 2\
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --best_n_size 20
```
* model_name_or_path: Directory to model or identifier for huggingface.co/models.EX:hfl/chinese-roberta-wwm-ext
* output_dir: Directory to checkpoints and trained model.
* train_file:Directory to train file .
* validation_file: Directory to validation file.
*context_file: Directory to context file.
*cache_dir:Directory to cache.
*max_seq_length:the maximum length for feature.
*num_train_epochs:numer of train epochs.
*warmup_ratio:warmup_steps=warmup_ratio*total_steps
*best_n_size:number of considered start/end position.

### Test

```shell
python QA_test.py \
  --model_name_or_path <model_name_or_path> \
  --test_file <test_file> \
  --context_file <context_file> \
  --output_file <output_file>
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --best_n_size 20
```

* model_name_or_path: Directory to model or identifier for huggingface.co/models.EX:hfl/chinese-roberta-wwm-ext
* test_file:Directory to test file .
*context_file: Directory to context file.
*output_file: Directory to output file.
*cache_dir:Directory to cache.
*max_seq_length:the maximum length for feature.
*n_best_size:number of considered start/end position.

### Reproduce my result (Public:0.80018,Private:0.80036)

```shell
bash download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```