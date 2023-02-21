import json
from tw_rouge import get_rouge
import os

result_path="./pred_data/"
json_name=os.listdir(result_path)
refs = []
public_path="./public.jsonl"
output_list=[]
rouge_1_list=[]
rouge_2_list=[]
rouge_l_list=[]

with open(public_path,encoding="utf8") as f:
    for line in f:
        line = json.loads(line)
        refs.append(line['title'].strip())
        #preds[line['id']] = line['title'].strip() + '\n'
        
for name in json_name:

    with open(result_path+name,encoding="utf8") as f:
        result = json.load(f)
    a=get_rouge(result['preds'], refs)
    rouge_1_list.append(a['rouge-1']['f'])
    rouge_2_list.append(a['rouge-2']['f'])
    rouge_l_list.append(a['rouge-l']['f'])
    b=json.dumps(a, indent=2)
    print(b)
    
answer_path="./"
with open(os.path.join(answer_path, "score.json"), "w") as f:
    json.dump({"r1": rouge_1_list,"r2": rouge_2_list,"rl": rouge_l_list}, f)
