import pandas as pd

df = pd.read_csv("./data/users/users.csv")

interests =set()
interests_list=[]
for index,row in df.iterrows():

    data = row['interests']
    if pd.isna(data):
        continue
    data_list = data.split(",")
    for interest in data_list:
         interests.update([interest])

while len(interests) != 0 :
    interest = interests.pop()
    interests_list.append(interest)
interest_id = [i for i in range(len(interests_list))]
df = pd.DataFrame(interests_list,columns=["interest_name"])
df['interest_id'] = interest_id

df.to_csv("./data/interests.csv",index=False)