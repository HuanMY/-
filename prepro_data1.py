import pandas as pd
from glob import glob
import re
import platform
import os
from collections import Counter
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def build_data():
	filepaths = glob("data/ruijin_round1_train2_20181022/*.txt")
	span = 300
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

	savepath = "data/data1.data"
	if os.path.exists(savepath):
		os.remove(savepath)
	lines = ""
	f = open(savepath,'a',encoding = "utf-8")
	for i in range(1,len(data)+1):
		if data[i-1]==" " or data[i-1] =="\u2003":
			d = "SPACE"
		elif data[i-1]=="\n":
			d = "LB"
		else:
			d = data[i-1]

		lines+=d+"\t"+label[i-1]+"\n"
		if i%span==0:
			lines+="\n"
	f.write(lines)
	f.close()
	print("length of data:",len(data))
def get_trainval(test_radio,shuffle):
	data = []
	label = []
	train_data = []
	train_label = []
	classes = ["O","B-Disease","B-Reason","B-Symptom","B-Test","B-Test_Value","B-Drug","B-Frequency","B-Amount","B-Method","B-Treatment","B-Operation","B-SideEff","B-Anatomy","B-Level","B-Duration","I-Disease","I-Reason","I-Symptom","I-Test","I-Test_Value","I-Drug","I-Frequency","I-Amount","I-Method","I-Treatment","I-Operation","I-SideEff","I-Anatomy","I-Level","I-Duration"]	
	with open("data/data1.data",'r',encoding = "utf-8") as fr:
		file = fr.readlines()
	for i in range(len(file)-1):
		if file[i]=="\n":
			label.append("\n")
			data.append("\n")
		else:
			file1 = file[i].strip().split("\t")
			if file1[0].isdigit():
				data.append("NUM")
			else:
				data.append(file1[0])
			label.append(file1[1])
	assert(len(data)==len(label))
	words_count = Counter(data)
	words = [word[0] for word in words_count.items() if word[1]>2]
	wordsids = dict(zip(words,np.arange(1,len(words)+1)))
	wordsids['UNK'] = len(words)+1
	wordsids['PAD'] = 0
	# start_index = [x.start() for x in re.finditer("\n","".join(data))]
	start_index = [x for x in range(len(data)) if data[x]=="\n"]
	start_index = [-1] +start_index+[len(data)]

	for i in range(len(start_index)-1):
		subdata = data[start_index[i]+1:start_index[i+1]]
		sublabel = label[start_index[i]+1:start_index[i+1]]
		if sublabel ==[]:
			continue

		train_data.append([wordsids.get(x,wordsids['UNK']) for x in subdata])
		train_label.append([classes.index(x) for x in sublabel])
	train_data = train_data[:-1]
	train_label = train_label[:-1]
	
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

	with open("data/wordsids_classes1.pkl",'wb') as fr:
		pickle.dump((wordsids,classes),fr)

	return train,val,train_label,val_label,val_,val_label_



def get_trainval_tf(test_radio,shuffle):
	data = []
	label = []
	train_data = []
	train_label = []
	classes = ["O","B-Disease","B-Reason","B-Symptom","B-Test","B-Test_Value","B-Drug","B-Frequency","B-Amount","B-Method","B-Treatment","B-Operation","B-SideEff","B-Anatomy","B-Level","B-Duration","I-Disease","I-Reason","I-Symptom","I-Test","I-Test_Value","I-Drug","I-Frequency","I-Amount","I-Method","I-Treatment","I-Operation","I-SideEff","I-Anatomy","I-Level","I-Duration"]	
	with open("data/data1.data",'r',encoding = "utf-8") as fr:
		file = fr.readlines()
	for i in range(len(file)-1):
		if file[i]=="\n":
			label.append("\n")
			data.append("\n")
		else:
			file1 = file[i].strip().split("\t")
			if file1[0].isdigit():
				data.append("NUM")
			else:
				data.append(file1[0])
			label.append(file1[1])
	assert(len(data)==len(label))
	words_count = Counter(data)
	words = [word[0] for word in words_count.items() if word[1]>=2]
	wordsids = dict(zip(words,np.arange(1,len(words)+1)))
	wordsids['UNK'] = len(words)+1
	wordsids['PAD'] = 0
	# start_index = [x.start() for x in re.finditer("\n","".join(data))]
	start_index = [x for x in range(len(data)) if data[x]=="\n"]
	start_index = [-1] +start_index+[len(data)]

	for i in range(len(start_index)-1):
		subdata = data[start_index[i]+1:start_index[i+1]]
		sublabel = label[start_index[i]+1:start_index[i+1]]
		if sublabel ==[]:
			continue

		train_data.append([wordsids.get(x,wordsids['UNK']) for x in subdata])
		train_label.append([classes.index(x) for x in sublabel])
	

	seq_len_list = np.array([len(x) for x in train_data])
	maxlen = max(seq_len_list)
	for i in range(len(train_data)):
		train_data[i] = train_data[i]+[wordsids['PAD']]*(maxlen-len(train_data[i]))
		train_label[i] = train_label[i]+[-1]*(maxlen-len(train_label[i]))
	train_data = np.array(train_data)
	train_label = np.array(train_label)	
	print("######### shape of train_label:",train_label.shape)
	l = int(len(train_data)*test_radio)
	index = np.arange(len(train_data))
	print("############## length of train :%d ,length of val:%d"%(len(train_data)-l,l))
	

	if shuffle:
		random.shuffle(index)
	train,val = train_data[index[:-l]],train_data[index[-l:]]
	train_label,val_label = train_label[index[:-l]],train_label[index[-l:]]
	seq_len_list,seq_len_list_val = seq_len_list[index[:-l]],seq_len_list[index[-l:]]

	with open("data/wordsids_classes1.pkl",'wb') as fr:
		pickle.dump((wordsids,classes),fr)

	return train,val,train_label,val_label,seq_len_list,seq_len_list_val



if __name__=="__main__":
	build_data()
	train,val,train_label,val_label,val_,val_label_ = get_trainval(0.1,False)







			



















