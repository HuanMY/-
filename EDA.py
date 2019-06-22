<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re

def length_of_sentence(data):
	length = [len(x) for x in data]
	plt.hist(length,bins = 50)
	plt.show()

def length_of_line():
	length = []
	filepaths = glob("data/ruijin_round1_train2_20181022/*.txt")
	for path in filepaths:
		with open(path,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
		length.extend([len(x) for x in file])
	print("length of length ",len(length))
	plt.hist(length,bins = 50)
	plt.show()

def build_data():
	filepaths = glob("data/ruijin_round1_train2_20181022/*.txt")
	line_len = 20
	span = 300
	data = []
	label = []
	flag_char = ['。','？',',',';','!','.']
	for datapath in filepaths:
		length_sentence = 0
		with open(datapath,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
			for i in range(len(file)):
				if len(file[i])>1 and len(file[i])<line_len and file[i][-2] not in flag_char:
					file[i] = file[i][:-1]+"。"

			file = "".join(file)
		labelpath = datapath[:-3]+"ann"
		lab = pd.read_table(labelpath,header = None,names = ["id","classes_position","words"])
		sublab = ['O']*len(file)
		for index,row in lab.iterrows():
			cla_p= row['classes_position'].split(" ")
			cla,p_s,p_e = cla_p[0],int(cla_p[1]),int(cla_p[-1])
			sublab[p_s] = "B-"+cla
			sublab[p_s+1:p_e] = ["I-"+cla]*(p_e-p_s-1)
		split_char = '。|;'
		index_split = [x.start()+1 for x in re.finditer(split_char,file)]
		index_split = [0] +index_split +[len(file)]
		for i in range(len(index_split)-1):
			s_e = [index_split[i],index_split[i+1]]
			if s_e[1]-s_e[0] <=span:
				data.append(file[s_e[0]:s_e[1]])
				label.append(sublab[s_e[0]:s_e[1]])
				length_sentence +=1

			else:
				for j in range((s_e[1]-s_e[0])//span+1):
					s = s_e[0]+j*span
					e = s+span
					if e>s_e[1]:
						e = s_e[1]
					if s==e:
						break
					data.append(file[s:e])
					label.append(sublab[s:e])
					length_sentence +=1
		assert(len(data)==len(label))
	return data
if __name__=="__main__":
	data = build_data()
	length_of_sentence(data)	








=======
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import jieba
import numpy as np
import pickle
jieba.load_userdict("Demo/DataSets/userdict.txt")
def inde(x,words_ind):
    try:
        return words_ind.index(x)+1
    except:
        for i in range(len(words_ind)):
            if words_ind[i]>=x:
                break
        return i

def word_len_two_class():
    length = []
    c = 0
    paths = glob("Demo/DataSets/ruijin_round2_train/ruijin_round2_train/*.txt")
    print("length of train:",len(paths))
    for index,path in enumerate(paths):
        print(index)
        with open(path,'r',encoding = "utf-8") as fr:
            content = fr.readlines()
            content = "".join(content)
        file = pd.read_table(path[:-3]+"ann",header = None)
        file_T = file[~file[2].isnull()]
        file_R = file[file[2].isnull()]

        file_T['classes'] = file_T[1].apply(lambda x:x.split(' ')[0])
        file_T['start'] = file_T[1].apply(lambda x:int(x.split(" ")[-1]))
        file_T['end'] = file_T[1].apply(lambda x:int(x.split(" ")[1]))

        file_R['relation'] = file_R[1].apply(lambda x:x.split(" ")[0])
        file_R['start'] = file_R[1].apply(lambda x:x.split(" ")[1].split(":")[1])
        file_R['end'] = file_R[1].apply(lambda x:x.split(" ")[2].split(":")[1])

        for ind,row in file_R.iterrows():
            relation = row['relation']
            start = row['start'] 
            end = row['end'] 
            f1 =  file_T[file_T[0]==start]
            f2 = file_T[file_T[0]==end]
            try:
                
                p = [f1['start'].iloc[0],f1['end'].iloc[0],f2['start'].iloc[0],f2['end'].iloc[0]]
            except:
                c+=1
            length.append(max(p)-min(p))
    print(max(length),min(length),c)    

def cla(x,flag):
    c = x.split("_")
    if len(c)==1:
        c = x.split("-")
    return c[flag-1]

def words_len_two_class():
    length = []
    c_err = 0
    rpl = "魑"
    classes = []
    classes_R = ['Test_Disease','Symptom_Disease','Treatment_Disease','Drug_Disease','Anatomy_Disease','Frequency_Drug','Duration_Drug','Amount_Drug','Method_Drug','SideEff-Drug']
    for x in classes_R:
        c = x.split("_")
        if len(c)==1:
            c = x.split("-")   
        if c[0] not in classes:
            classes.append(c[0])
        if c[1] not in classes:
            classes.append(c[1])
         
    paths = glob("Demo/DataSets/ruijin_round2_train/ruijin_round2_train/*.txt")
#     paths = ['1.txt']
    print("length of train:",len(paths))
    for index,path in enumerate(paths):
        print(index)
        with open(path,'r',encoding = "utf-8") as fr:
            file = fr.readlines()
            file = "".join(file)
            file = file.replace("\n",rpl)        
        ann = pd.read_table(path[:-3]+"ann",names = ['id','class_position','content'],header = None)
        ann_T = ann[~ann['content'].isnull()]
        ann_R = ann[ann['content'].isnull()]

        ann_T['cla'] = ann_T['class_position'].apply(lambda x:x.split(" ")[0])
        ann_T['start'] = ann_T['class_position'].apply(lambda x:int(x.split(' ')[1]))
        ann_T['end'] = ann_T['class_position'].apply(lambda x:int(x.split(" ")[-1]))
        ann_R['cla'] = ann_R['class_position'].apply(lambda x:x.split(" ")[0])
        ann_R['start'] = ann_R['class_position'].apply(lambda x:x.split(" ")[1].split(":")[1])
        ann_R['end'] = ann_R['class_position'].apply(lambda x:x.split(" ")[2].split(":")[1]) 
        ann_R['cla1'] = ann_R['cla'].apply(lambda x:cla(x,1))
        ann_R['cla2'] = ann_R['cla'].apply(lambda x:cla(x,2))

        ann_T_c = ann_T[ann_T['cla'].isin(classes)]

        words = list(jieba.cut(file,HMM=True))
        words_ind = [len(x) for x in words]
        words_ind = [sum(words_ind[:i]) for i in range(1,len(words_ind))]
        ann_T_c['ind'] = ann_T_c['start'].apply(lambda x:inde(x,words_ind))        

        for ind1,row in ann_R.iterrows():
            start = row['start'] 
            end = row['end']
            f1 =  ann_T_c[ann_T_c['id']==start]
            f2 = ann_T_c[ann_T_c['id']==end]
            try:
                p = [f1['ind'].iloc[0],f2['ind'].iloc[0]]
            except:
                c_err+=1
                continue      
            if min(p)!=max(p):
                length.append(max(p)-min(p))

        print("length of words:",len(words),max(length))                    

    print(max(length),min(length))
    print("c_err:",c_err)
    
    

    
    
if __name__ =="__main__":
    with open("Demo/DataSets/userword.pkl",'rb') as fr:
        userword = pickle.load(fr)
    with open("Demo/DataSets/data.pkl",'rb') as fr:
        file = pickle.load(fr)
    
    
    



>>>>>>> 0379ce00a52098dcbc0785951067d1eef1de64b0
