import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from re import I
from typing import Dict
import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from tqdm import trange
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.nn import Embedding
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
train_all_loss=[]
valid_all_loss=[]
test_mode=False


def save_checkpoint(model,epoch,ckpt_dir):
  best_ckpt_path=ckpt_dir/"best_model.pth"
  torch.save(model.state_dict(),best_ckpt_path)
  print("renew best weight at epoch:"+str(epoch))

def main(args):
    best_acc=float(0)
    best_valid_loss=10000000
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    a=torch.load("./cache/intent/embeddings.pt")
    embedding=Embedding.from_pretrained(a, freeze=False)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    for name in datasets:

        dataset=datasets.get(name)
        text_list=[col['text'].split(" ") for col in dataset]
        text_list_2=datasets["train"].vocab.encode_batch(text_list,dataset.max_len)

        for i in range(len(dataset.data)):#轉data到數字
            dataset.data[i]['text']=torch.LongTensor(text_list_2[i])
            dataset.data[i]['intent']=dataset.label_mapping[dataset.data[i]['intent']]

    # TODO: create DataLoader for train / dev datasets
    Train_loader=Data.DataLoader(dataset=datasets['train'],batch_size=args.batch_size,shuffle=True,num_workers=2)
    dev_loader=Data.DataLoader(dataset=datasets['eval'],batch_size=args.batch_size,shuffle=True,num_workers=2)
    embeddings = torch.load(args.cache_dir / "embeddings.pt") #6491*300
    # TODO: init model and move model to target device(cpu / gpu)
    if torch.cuda.is_available():
        device_id="cuda:"+str(torch.cuda.current_device())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        test_mode
    )
    model.to(device)
    print(model)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    if args.scheduler=="onecycle":
      scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(Train_loader), pct_start=0.2)
    elif args.scheduler == 'step':
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.scheduler == 'reduce':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    else:
      scheduler = None
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    iter=0
    for epoch in epoch_pbar:
        print(" ")
        print_every=5000 #5000筆data print一次loss
        plot_every=1000
        current_loss=0
        train_epoch_loss=0
        last_loss=0
        train_num=0
        # TODO: Training loop - iterate over train dataloader and update model weights
        for batch in Train_loader:
            model.train(mode=True)#啟用訓練模式
            iter=iter+100
            batch['text']=batch['text'].to(device)
            batch['intent']=batch['intent'].to(device)
            optimizer.zero_grad() #zero gradients for every batch
            output=model(batch) #outputs為預測機率  
            loss=loss_fn(output['prob'],batch['intent'])
            loss.backward() #一但被backward optimizer就知道其值
            optimizer.step()
            if args.scheduler == "onecycle":
              scheduler.step()
            for i in range(len(batch['intent'])):
              if output['indice'][i]==batch['intent'][i]:
                train_num=train_num+1
                #print("正確:"+str(indices[i]))
            #print("正確率:"+str(num/len(indices)))
            # Gather data and report
            train_epoch_loss += loss.item()
            current_loss += loss.item()
            if iter % plot_every==0:
                train_all_loss.append(current_loss/plot_every)
                train_current_loss=0
            #if(iter%print_every==0):
                #print("Epoch:"+str(epoch)+",s目前處理data量:"+str(iter)+"loss:"+str(loss.item()))
        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_num=0
        valid_epoch_loss=0
        valid_current_loss=0
        for batch in dev_loader:
          model.eval()#啟用validation模式
          batch['text']=batch['text'].to(device)
          batch['intent']=batch['intent'].to(device)
          output=model(batch) 
          loss=output['loss']
          for i in range(len(batch['intent'])):
            if output['indice'][i]==batch['intent'][i]:
              valid_num=valid_num+1
          valid_epoch_loss += loss.item()
          valid_current_loss += loss.item()
          if iter % plot_every==0:
              valid_all_loss.append(current_loss/plot_every)
              valid_current_loss=0
          #if(iter%print_every==0):
              #print("Epoch:"+str(epoch)+",s目前處理data量:"+str(iter)+"loss:"+str(loss.item()))
        print(" ")
        print("-----------------------------"+"epoch:"+str(epoch)+"---------------------------------")
        print("Train Loss: {:.4f}".format(train_epoch_loss))
        print("Train_data正確率:"+str(train_num/len(Train_loader.dataset))) 
        print("Valid Loss: {:.4f}".format(valid_epoch_loss))
        val_accuracy=valid_num/len(dev_loader.dataset)
        print("Valid_data正確率:"+str(val_accuracy))
        print("-----------------------------"+"epoch:"+str(epoch)+"---------------------------------")
        print("best acc:"+str(best_acc))
        print("best_loss:"+str(best_valid_loss))
        if val_accuracy>best_acc:
          best_acc=val_accuracy
          save_checkpoint(model,epoch,args.ckpt_dir)
        elif valid_epoch_loss<best_valid_loss:
          best_valid_loss=valid_epoch_loss
          file_name="best_intent.pth"
          ckpt_path=args.ckpt_dir/file_name
          torch.save(model.state_dict(),ckpt_path)
        if (epoch+1)/5==0:
          ckpt_path=args.ckpt_dir/"epoch_{}_model.pth".format(epoch+1)
          torch.save(model.state_dict(),ckpt_path)
    print("best_acc:"+str(best_acc))
    print("best_loss:"+str(best_valid_loss))
    path = 'setting.txt'
    f = open(path, 'w')

    f.write(str("hidden size:"+str(args.hidden_size))+"\n")
    f.write(str("dropout:"+str(args.dropout))+"\n")
    f.write(str("num layers:"+str(args.num_layers))+"\n")
    f.write(str("learning rate:"+str(args.lr))+"\n")
    f.write(str("num epoch:"+str(args.num_epoch))+"\n")
    f.write(str("bidirectional:"+str(args.bidirectional))+"\n")
    f.write(str("max len:"+str(args.max_len))+"\n")
    f.write(str("batch size:"+str(args.batch_size))+"\n")
    f.write("valid score:"+str(best_acc)+"\n")
    f.write("valid loss:"+str(best_valid_loss)+"\n")
          


          


    #plt.figure()
    #plt.plot(train_all_loss)
    #plt.plot(valid_all_loss)
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    parser.add_argument("--scheduler",help="'type:MultiStepLR,OneCycleLR,StepLR,ReduceLROnPlateau,ExponentialLR,CosineAnnealingLR,LambdaLR",default='onecycle', type=str)

    parser.add_argument("--step_size", type=int, default=15)

    parser.add_argument("--weight_decay",help="regularization",default='0.01', type=str)

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
