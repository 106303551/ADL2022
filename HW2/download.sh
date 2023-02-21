python download_pretrained_model.py --model_name_or_path hfl/chinese-roberta-wwm-ext --Question_type MC --cache_dir ./cache/ 
python download_pretrained_model.py --model_name_or_path hfl/chinese-roberta-wwm-ext-large --Question_type QA --cache_dir ./cache/ 

wget https://www.dropbox.com/s/srs3030zav89ko3/pytorch_model.bin?dl=1 -O ./model/MC/pytorch_model.bin
wget https://www.dropbox.com/s/e5slxn3w46eqcz6/pytorch_model.bin?dl=1 -O ./model/QA/pytorch_model.bin
     