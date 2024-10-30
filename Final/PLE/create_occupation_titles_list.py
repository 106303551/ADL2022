import pandas as pd

df = pd.read_csv("./data/users/users.csv")

recreations =set()
interests_list=[]
for index,row in df.iterrows():

    data = row['occupation_titles']
    if pd.isna(data):
        continue
    data_list = data.split(",")
    for recreation in data_list:
         recreations.update([recreation])

while len(recreations) != 0 :
    interest = recreations.pop()
    interests_list.append(interest)
interest_id = [i for i in range(len(interests_list))]
df = pd.DataFrame(interests_list,columns=["name"])
df['id'] = interest_id

df.to_csv("./data/occupation_titles.csv",index=False)