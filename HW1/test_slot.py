import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch.utils.data as Data
import torch
import pandas as pd
from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
test_mode=True

def main(args):

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    token_list=[col['tokens'] for col in dataset]
    len_list=[]
    for i in range(len(token_list)):
      token_len=len(token_list[i])
      len_list.append(token_len)
    token_list_2=dataset.vocab.encode_batch(token_list,dataset.max_len)
    for i in range(len(dataset.data)):#轉data到數字
        dataset.data[i]['tokens']=torch.LongTensor(token_list_2[i])
        dataset.data[i]['len']=len_list[i]
        dataset.data[i].pop('tags',None)
        #dataset.data[i]['intent']=dataset.label_mapping[dataset.data[i]['intent']] #之後要刪掉
        #print(dataset.data[i]['tokens'])
    # TODO: crecate DataLoader for test dataset
    print(dataset.data[0])
    Test_loader=Data.DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=False,num_workers=2)
    if torch.cuda.is_available():
        device_id="cuda:"+str(torch.cuda.current_device())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    embeddings = torch.load(args.cache_dir / "embeddings.pt").to(device)
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        test_mode,
    )
    model.to(device)
    print(model)
    model.eval()

    ckpt = torch.load(args.ckpt_path,map_location=device)
    # load weights into model
    model.load_state_dict(ckpt)
    # TODO: predict dataset
    train_num=0
    sum=0
    j=0
    id_list=[]
    label_list=[]
    i=-1
    for batch in Test_loader:
      i=i+1
      j=j+1
      batch['tokens']=batch['tokens'].to(device)
      output=model(batch)
      for i in range(len(output['out_tag_list'])):
        label=""
        out_tag_list=output['out_tag_list'][i]
        for j in range(len(output['out_tag_list'][i])):  
          label=label+dataset.idx2label(out_tag_list[j])+" "
        label=label.rstrip()
        label_list.append(label)
        print(batch['id'][i])
        id_list.append(batch['id'][i])
    #   for i in range(len(output['indice'])):
    #     sum=sum+1
    #     if output['indice'][i]==batch['intent'][i]:
    #       train_num=train_num+1
    #   print("準確率:"+str(train_num/sum))
    # print("總準確率:"+str(train_num/len(Test_loader.dataset)))
    # TODO: write prediction to file (args.pred_file)
    frame=pd.DataFrame({'id':id_list})
    frame=frame.assign(tags=label_list)
    frame.to_csv(args.pred_file,index=False)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="best_slot.pth"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--test_mode",type=bool,default=True)
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)