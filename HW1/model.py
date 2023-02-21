from re import X
from typing import Dict, List
from torch.nn import functional as F
import torch
from torch.nn import Embedding
import torch.nn as nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int, #150個intent
        test_mode: bool
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False) 
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.num_class=num_class
        self.dropout=dropout
        self.test_mode=test_mode
        # TODO: model architecture
        self.rnn=torch.nn.LSTM(embeddings.shape[1],self.hidden_size,self.num_layers,batch_first=True,bidirectional=self.bidirectional,dropout=self.dropout)
        self.lin=torch.nn.Linear(self.encoder_output_size,self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn bidirectional會影響outputsize
        if self.bidirectional==True:
          return 2*self.hidden_size
        else:
          return self.hidden_size


    def forward(self, batch) -> Dict[str, torch.Tensor]:

        # TODO: implement model forward
        output={}
        train_x=batch['text'] # batch_size*max_len(128)
        train_x=self.embed(train_x)
        out,(h,_)=self.rnn(train_x)  #output=out,(h_n,c_n) for lstm out=(batch_size,seq_len,hidden_size * num_directions) o_t
                                    #out=contain the output features(h_t)from last layer of LSTM,for each t.
                                    #h_n=final hidden state for each element in the sequence.concatenation of final forward and reverse hidden states
                          #c_n=傳到下個cell的hidden_state輸出
        if self.bidirectional:
          
          out_0_back=out[:,0,self.hidden_size:len(out[0,0])] #作為反向時為最後一層,
          out_1_front=out[:,-1,0:self.hidden_size] #作為順向時為最後一層
          # out_tensor = torch.tensor([])
          # for i in range(len(out_0_back)):
          #   out_0=out_0_back[i]
          #   out_1=out_1_front[i]
            # out_i=torch.cat(out_0,out_1)
        out=torch.cat((out_1_front,out_0_back),dim=1)

        out_prob = self.lin(out) #128*128*150 此為機率值
        values,indices=torch.max(out_prob,1)
        if self.test_mode==False:
          loss_fn = torch.nn.CrossEntropyLoss()
          loss=loss_fn(out_prob,indices)
          output['loss']=loss
          
        #將資訊存入output dict
        output['text']=train_x
        output['prob']=out_prob
        output['indice']=indices
        return output

class SeqTagger(SeqClassifier):
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        #TODO: implement model forward
        output={}
        train_x=batch['tokens'] # batch_size*max_len(128)
        train_x=self.embed(train_x)#128*128*300
        out,_=self.rnn(train_x)  #output=out,(h_n,c_n) for lstm out=(batch_size,seq_len,hidden_size * num_directions) o_t
                                  #h_n=當下lstmcell的輸出
                          #c_n=傳到下個cell的hidden_state輸出
        out_prob = self.lin(out) #(num,t,tag_class)=128*128*9 
        a=out_prob.max(-1,keepdim=True)
        v=out_prob.max(-1,keepdim=True)[1] #num*128*1
        values,indices=torch.max(out_prob,2) #(num,t,tags)=num*128
        if torch.cuda.is_available():
          device_id="cuda:"+str(torch.cuda.current_device())
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
          device=torch.device("cpu")
        out_tag_list=[]
        prob_tensor=torch.empty(0).to(device)
        #train用
        com_label_list=[]
        label_list=[]
        
        for i in range(out_prob.shape[0]):
            prob_tensor=torch.cat((prob_tensor,out_prob[i,0:batch['len'][i]]),0)
            tag_i=indices[i,0:batch['len'][i]].tolist()
            out_tag_list.append(tag_i)#預測的tag
            if self.test_mode==False:
              label_i=batch['tags'][i][0:batch['len'][i]].tolist()
              label_list=label_list+label_i
              com_label_list.append(label_i)#正確的tag
        
        if self.test_mode==False:
          label=torch.tensor(label_list).to(device)
          loss_fn = torch.nn.CrossEntropyLoss()
          loss=loss_fn(prob_tensor,label)

        #將資訊存入output dict
        output['out_tag_list']=out_tag_list
        output['text']=train_x
        output['prob']=out_prob
        output['tags']=indices
        if self.test_mode==False:
          output['tag_list']=com_label_list
          output['loss']=loss
        
        return output

