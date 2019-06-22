import pandas as pd
from glob import glob
import re
import platform
import os
from collections import Counter
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def get_trainval(test_radio,shuffle):
	filepaths = glob("data/ruijin_round1_train2_20181022/*.txt")
	span = 400
	data = []
	label = []
	for datapath in filepaths:
		length_sentence = 0
		with open(datapath,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
			file = "".join(file)
		labelpath = datapath[:-3]+"ann"
		lab = pd.read_table(labelpath,header = None,names = ["id","classes_position","words"])
		sublab = ['O']*len(file)
		for index,row in lab.iterrows():
			cla_p= row['classes_position'].split(" ")
			cla,p_s,p_e = cla_p[0],int(cla_p[1]),int(cla_p[-1])
			sublab[p_s] = "B-"+cla
			sublab[p_s+1:p_e] = ["I-"+cla]*(p_e-p_s-1)
		data.extend(list(file))
		label.extend(sublab)

	for i in range(len(data)):
		if data[i]==" " or data[i] =="\u2003":
			data[i] = "SPACE"
		elif data[i]=="\n":
			data[i] = "LB"
		elif data[i].isdigit():
			data[i] = "NUM"


	train_data = []
	train_label = []
	classes = ["O","B-Disease","B-Reason","B-Symptom","B-Test","B-Test_Value","B-Drug","B-Frequency","B-Amount","B-Method","B-Treatment","B-Operation","B-SideEff","B-Anatomy","B-Level","B-Duration","I-Disease","I-Reason","I-Symptom","I-Test","I-Test_Value","I-Drug","I-Frequency","I-Amount","I-Method","I-Treatment","I-Operation","I-SideEff","I-Anatomy","I-Level","I-Duration"]	

	words_count = Counter(data)
	words = [word[0] for word in words_count.items() if word[1]>2]
	wordsids = dict(zip(words,np.arange(1,len(words)+1)))
	wordsids['UNK'] = len(words)+1
	wordsids['PAD'] = 0


	data = [wordsids.get(x,wordsids['UNK']) for x in data]
	label = [classes.index(x) for x in label]
	


	# start_index = [x.start() for x in re.finditer("\n","".join(data))]
	start_index = [x for x in range(len(data)) if x%(span//2)==0]
	for s_i in start_index:
		train_data.append(data[s_i:s_i+span])
		train_label.append(label[s_i:s_i+span])
	train_data  = [x for x in train_data if len(x)==span]
	train_label = [x for x in train_label if len(x)==span]


	l = int(len(train_data)*test_radio)
	temp_val = train_data[-l:]
	temp_label = train_label[-l:]
	val_,val_label_ = [],[]
	[val_.extend(x) for x in temp_val]
	[val_label_.extend(x) for x in temp_label]
	val_,val_label_ = np.array(val_).reshape((1,len(val_))),np.array(val_label_).reshape((1,len(val_)))

	# train_data = pad_sequences(train_data,maxlen = None,value = wordsids['PAD'])
	# train_label = pad_sequences(train_label,maxlen = None,value = -1)
	train_data = np.array(train_data)
	train_label = np.array(train_label)	
	train_label = np.expand_dims(train_label,2)	
	print("######### shape of train_labe:{},lengtho of wordsids:{}".format(train_label.shape,len(wordsids)))
	l = int(len(train_data)*test_radio)
	index = np.arange(len(train_data))
	print("############## length of train :%d ,length of val:%d"%(len(train_data)-l,l))
	

	if shuffle:
		random.shuffle(index)
	train,val = train_data[index[:-l]],train_data[index[-l:]]
	train_label,val_label = train_label[index[:-l]],train_label[index[-l:]]


	return train,val,train_label,val_label,val_,val_label_,wordsids,classes






if __name__=="__main__":
	train,val,train_label,val_label,val_,val_label_,wordsids,classes = get_trainval(0.1,False)







			



















