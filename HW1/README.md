# HW 1 

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection

### Train
```shell
python train_intent.py -data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout>  --max_len <max_len> --num_epoch <num_epoch>
```
* data_dir: Directory to dataset.
* cache_dir: Directory to the preprocessed caches.
* ckpt_dir: Directory to save the model file.
* hidden_size: LSTM hidden state dimension. default:512
*num_layers:Number of layers. default:2
*batch_size:number of data for batch.default:128
*dropout:Model's dropout rate default:0.5
*num_epoch:Number of epoch. default:50

### Predict

```shell
python test_intent.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> 
```

* test_file: Path to the test file.
* cache_dir: Directory to the preprocessed caches.
* ckpt_path: Path to model checkpoint.
* pred_file: Predict file save path. default:pred.intent.csv
* hidden_size: LSTM hidden state dimension. default:512
* num_layers: Number of layers. default:2
* dropout: Model's dropout rate. default:0.5

### Reproduce my result (Public:0.91866,Private:)

```shell
bash download.sh
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```
### Slot tagging

### Train
```shell
python train_slot.py -data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout>  --max_len <max_len> --num_epoch <num_epoch>
```
* data_dir: Directory to dataset.
* cache_dir: Directory to the preprocessed caches.
* ckpt_dir: Directory to save the model file.
* hidden_size: LSTM hidden state dimension. default:512
*num_layers:Number of layers. default:2
*batch_size:number of data for batch.default:128
*dropout:Model's dropout rate default:0.5
*num_epoch:Number of epoch. default:50

### Predict

```shell
python test_intent.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> 
```

* test_file: Path to the test file.
* cache_dir: Directory to the preprocessed caches.
* ckpt_path: Path to model checkpoint.
* pred_file: Predict file save path. default:pred.intent.csv
* hidden_size: LSTM hidden state dimension. default:512
* num_layers: Number of layers. default:2
* dropout: Model's dropout rate. default:0.5

### Reproduce my result (Public:0.91866,Private:)

```shell
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```


