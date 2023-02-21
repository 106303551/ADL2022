

python MC_test.py \
  --model_name_or_path ./model/MC/ \
  --test_file $2 \
  --context_file $1 \
  --output_file ./result.json \
  --cache_dir ./cache/ \
  --max_seq_length 512 \

python QA_test.py \
  --model_name_or_path ./model/QA/ \
  --test_file ./result.json \
  --context_file $1 \
  --output_file $3 \
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --n_best_size 20 \