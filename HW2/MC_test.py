import pandas
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datasets import DatasetDict
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers import AutoTokenizer
from itertools import chain
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]

def parse_args() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Directory to model or identifier for huggingface.co/models.",
        default="./model/QA"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Directory to the test file.",
        default="./data/test.json"
    )
    parser.add_argument(
        "--context_file",
        type=str,
        help="Directory to the context.",
        default="./data/context.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Directory to the output_file.",
        default="./result.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to the cache.",
        default="./cache/"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="the maximum length for feature.",
        default=512
    )
    args = parser.parse_known_args()[0]
    return args

args=parse_args()
ds=DatasetDict.from_json({'test':args.test_file})
with open(args.context_file,encoding="utf-8") as f:
    context=json.load(f)

model_checkpoint=args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint,cache_dir=args.cache_dir,)

#將context加入dict
para_dict={'1':[],'2':[],'3':[],'4':[]}
label=[]
for i in range(4):
  for j in range(len(ds['test'])):
    para_dict[str(i+1)].append(context[ds['test'][j]['paragraphs'][i]])
  ds['test'] = ds['test'].add_column("para_"+str(i+1), para_dict[str(i+1)])

def preprocess_function(examples):
    ending_names = ["para_1", "para_2", "para_3", "para_4"]
    first_sentences = [[context] * 4 for context in examples["question"]]
    #question_headers = examples[question_header_name]
    second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i in range(len(examples['question']))]
    #labels = examples["label"]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=512,
        padding=True,
        truncation=True,
    )
    # Un-flatten
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    #tokenized_inputs["labels"] = labels
    return tokenized_inputs

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def __call__(self, features):
        #label_name = "label"
        #labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        #batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        #for k,v in batch.items():
          #batch[k]=v.to(device)
        return batch



def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

model_checkpoint="bert-base-chinese"
model_name = model_checkpoint.split("/")[-1]
batch_size=2
train_args = TrainingArguments(
    output_dir=f"./{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model,
    train_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    
)

encoded_datasets=ds.map(preprocess_function, batched=True)
results=trainer.predict(encoded_datasets['test'])

pred=np.argmax(results.predictions,axis=1)
ds['test'] = ds['test'].add_column("label",pred)
ds=ds.map(remove_columns=['para_1','para_2','para_3','para_4'])

with open(args.output_file, "w") as outfile:
  dic_list=[]
  for i in range(len(pred)):
    diction={
        'id':ds['test']['id'][i],
        'question':ds['test']['question'][i],
        'paragraphs':ds['test']['paragraphs'][i],
        'label':ds['test']['label'][i],
    }
    dic_list.append(diction)
  json_object = json.dumps(dic_list, indent=4,ensure_ascii=False)
  outfile.write(json_object)