# %%
import pandas as pd
import numpy as np
from keras.preprocessing import text,sequence
import json

######################处理得到sku_name: 三级品类 的dataframe########
########三级品类的数据量需要大于3000######
all_info=pd.read_csv("embed_eval_data_allinfo.csv")
all_info["sku_name"] = all_info["sku_name"].map(lambda x:str(x))
all_info["item_first_cate_name"] = all_info["item_first_cate_name"].map(lambda x:str(x))
all_info = all_info [all_info["sku_name"] != "nan"]
all_info = all_info[all_info["item_first_cate_name"] != "nan"]
all_info["item_cate"] = all_info["item_first_cate_name"]+"@@"+all_info["item_second_cate_name"]+"@@"+all_info["item_third_cate_name"]
cate_num = dict(all_info["item_cate"].value_counts())
cate = all_info.drop_duplicates(["item_cate"])
cate = list(cate["item_cate"])
cate2id = {}
id2cate = {}
id = 0
for i in cate:
    if "测试" not in i and cate_num[i] > 3000:
        print(i)
        cate2id[i] = id
        id2cate[id] = i
        id += 1

json.dump(id2cate,open("id2cate.json","w"))

######### 计算label的两两相似度：greedy search, Jaccard sim #######
'''
cate_sim_matrix = np.ones((len(cate2id),len(cate2id)))
for i in range(len(cate2id)):
    for j in range(i):
        print(i," ",i/len(cate2id)," ",len(cate2id))
        print("j ",j)
        print("similarity between ", id2cate[i], "and ",id2cate[j])
        textA = all_info[all_info["item_cate"] == id2cate[i]]["sku_name"]
        textB = all_info[all_info["item_cate"] == id2cate[j]]["sku_name"]
        wordA = []
        wordB = []
        for k in textA:
            wordA.extend(k.strip().split(" "))
        for k in textB:
            wordB.extend(k.strip().split(" "))
        A_s = set(wordA)
        B_s = set(wordB)
        jac_sim = len(A_s.intersection(B_s)) / len(A_s.union(B_s))
        print(jac_sim)
        cate_sim_matrix[i][j] = jac_sim
np.save("cate_sim_matrix",cate_sim_matrix)
'''
########### 先取矩阵中最相似的两个label############
id2cate = json.load(open("id2cate.json"))
cate_sim_matrix = np.load("cate_sim_matrix.npy")

a= np.unravel_index(cate_sim_matrix.argmin(),cate_sim_matrix.shape)
chosen_cateid = [a[0], a[1]]
print(chosen_cateid)
####### greedy search#########
while(len(chosen_cateid)<10):
    cate_sim = np.ones(len(id2cate)) ## 每个category到已经选择的categories的最大相似度
    for i in range(len(id2cate)): ##遍历每个类别
        chosen_sim = []  ##类别i到已选择的类别的相似度
        for j in chosen_cateid:
            if i > j:
                sim = cate_sim_matrix[i][j] 
            if i == j:
                sim = 1
            if i < j:
                sim = cate_sim_matrix[j][i]
            chosen_sim.append(sim)
        chosen_sim = np.array(chosen_sim)
        max_sim = np.max(chosen_sim) ##
        cate_sim[i] = max_sim
    new_id = np.argmin(cate_sim)
    chosen_cateid.append(new_id)
label = {}
cnt = 0
data = None
for i in chosen_cateid:
    label[id2cate[str(i)]] = cnt
    cnt += 1
    print(id2cate[str(i)])
    df= all_info[all_info["item_cate"] == id2cate[str(i)]].sample(3000)[["sku_name","item_cate"]]
    if data is None:
        data = df
    else:
        data = data.append(df)


data.columns = ["sku_name","label"]
print(data)
json.dump(label,open("label.json","w"))
data.to_csv("embed_eval_data_sample_general.csv")
names = list(data["sku_name"])
out = open("sku_names.txt","w",encoding='utf-8')
for i in names:
    out.write(str(i)+"\n")
