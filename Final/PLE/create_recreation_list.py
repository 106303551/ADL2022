import pandas as pd

df = pd.read_csv("./data/users/users.csv")

recreations =set()
recreations_list=[]
for index,row in df.iterrows():

    data = row['recreation_names']
    if pd.isna(data):
        continue
    data_list = data.split(",")
    for recreation in data_list:
         recreations.update([recreation])

while len(recreations) != 0 :
    recreation = recreations.pop()
    recreations_list.append(recreation)
recreations_id = [i for i in range(len(recreations_list))]
df = pd.DataFrame(recreations_list,columns=["name"])
df['id'] = recreations_id

df.to_csv("./data/recreations.csv",index=False)