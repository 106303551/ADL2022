# HW 3

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw3"
make
conda activate adl-hw3
pip install -r requirements.txt
```



## Summarization

### Train
```shell
python MC_train.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --cache_dir ./cache/ \
  --max_length 256 \
  --max_target_length 64 \
  --learning_rate 2e-3 \
  --num_train_epochs 10 \
  --warmup_ratio 0.1 \
  --top_p 0.5 \
  --top_k 0 \
  --num_beams 1 \
  --do_sample True \
  --temperature 1 \
```
* model_name_or_path: Directory to model or identifier for huggingface.co/models. EX:google/mt5-small
* output_dir: Directory to checkpoints and trained model.
* train_file:Directory to train file .
* validation_file: Directory to validation file.
* cache_dir:Directory to cache.
* max_length:The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,equences shorter will be padded.
* max_target_length:The maximum total sequence length for target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.
* num_train_epochs:numer of train epochs.
* warmup_ratio: warmup_steps=warmup_ratio * total_steps
* do_sample:do sample for summary or not.
* top_p:uppper bound of acumulative probability.
* top_k:choose top k highest probability of token to decide token.
* temperature:設定temperature(T)改變probability distribution，在T較高時會讓probability distribution更加平緩，讓選字結果more diversity。在T較低時讓probability distribution更加spiky，讓選字結果less diversity。

### Test

```shell
python MC_test.py \
  --model_name_or_path <model_name_or_path> \
  --test_file <test_file> \
  --output_file <output_file> \
  --cache_dir ./cache/ \
  --max_target_length 64 \ 
```

* model_name_or_path: Directory to model or identifier for huggingface.co/models. EX:google/mt5-small
* test_file:Directory to test file .
* output_file: Directory to output file.
* cache_dir:Directory to cache.
* max_target_length:The maximum total sequence length for target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.





### Reproduce my result

```shell
bash download.sh
bash ./run.sh  /path/to/test.json  /path/to/pred/prediction.json
```