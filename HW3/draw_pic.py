from matplotlib import pyplot as plt
import json
import os
result_path="./score/"
json_name=os.listdir(result_path)
for name in json_name:
    with open(result_path+name,encoding="utf8") as f:
        result = json.load(f)
fig,ax = plt.subplots(1,1)
column_labels=["R1-score","R2-score","RL-score"]
data = [[result['r1'],result['r2'],result['rl']]]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")

plt.show()