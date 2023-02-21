python MC_test.py \
  --model_name_or_path ./model/ \
  --test_file $1 \
  --output_file $2 \
  --cache_dir ./cache/ \
  --num_beams 5 \
  --max_target_length 64 \ 