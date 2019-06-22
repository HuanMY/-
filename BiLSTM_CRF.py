import pandas as pd
from glob import glob
import numpy as np
import pickle
import os
import re
import tensorflow as tf
import time
import random
import os
import datetime
from sklearn.metrics import classification_report
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def Word2id():

	filepaths = glob("ruijin_round1_train2_20181022/*.txt")
	filepaths +=glob("ruijin_round1_test_a_20181022/*.txt")
	all_file = ""
	for filepath in filepaths:
		with open(filepath,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
		all_file += "".join(file)
	all_file =list(set(list(all_file)))
	for x in range(len(all_file)):
		if all_file[x].isdigit():
			all_file[x] = "NUM"

	all_file = list(set(all_file))
	allfile_dict = dict(zip(all_file,np.arange(1,len(all_file)+1)))
	with open("data/word2id.pkl",'wb') as fr:
		pickle.dump(allfile_dict,fr)
	# print("size of word: ",len(all_file))
	return allfile_dict
def tag2label():
	classes = ["O","Disease","Reason","Symptom","Test","Test_Value","Drug","Frequency","Amount","Method","Treatment","Operation","SideEff","Anatomy","Level","Duration"]
	classes =classes[:1]+ ["B-"+classes[i] for i in range(1,len(classes))]+["I-"+classes[i] for i in range(1,len(classes))]
	tag = dict(zip(classes,np.arange(len(classes))))
	tag_num = dict(zip(np.arange(len(classes)),classes))
	return tag,tag_num
def getlabel(labelpath,size):
	tag,_ = tag2label()	
	label = pd.read_table(labelpath,header=None,names = ["id","classes_position","words"])
	label['classes'] = label['classes_position'].apply(lambda x:x[:x.find(" ")])
	label['position'] = label['classes_position'].apply(lambda x:x[x.find(" ")+1:])		
	createlabel = np.zeros(size).tolist()
	for index,row in label.iterrows():
		position = row['position'].split(";")
		position_s = eval(position[0].split(" ")[0])
		position_e = eval(position[-1].split(" ")[-1])
		classes = row['classes']
		classes_s = "B-"+classes
		classes_e = "I-"+classes
		createlabel[position_s] = tag[classes_s]
		createlabel[position_s+1:position_e] = [tag[classes_e]]*(position_e-position_s-1)
	return createlabel	
def getfileid(file):
	word2id = Word2id()
	# file = [word2id[x] for x in file]
	file1 = []
	for x in file:
		if x.isdigit():
			file1.append(word2id['NUM'])
		else:
			file1.append(word2id[x])

	return file1

def padding(data,max_len):

	seq_len_list = [len(x) for x in data]
	# max_len = max(seq_len_list)
	for i in range(len(data)):
		data[i] = data[i]+[0]*(max_len-len(data[i]))
	return data,seq_len_list



def data_preprocessing(filepaths,testflag):
	train = []
	n = 300
	max_len =n

	if testflag ==0:
		train_label = []		
	for index2,filepath in enumerate(filepaths):
		if testflag ==0:
			labelpath = filepath.split("/")
			labelpath = os.path.join(labelpath[0],labelpath[1].split(".")[0]+".ann")
		with open(filepath,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
		file = "".join(file)
		data = getfileid(file)
		if testflag==0:
			label = getlabel(labelpath,len(file))
		index = [x.start() for x in  re.finditer("。|;",file)]
		# index = [-1]+index+[len(file)]
		index = [0]+index+[len(file)]


		for i in range(len(index)-1):
			index1 = [index[i],index[i+1]]
			
			if index1[1]-index1[0]<=300:
				train.append(data[index1[0]:index1[1]])
				if testflag==0:
					train_label.append(label[index1[0]:index1[1]])
			else:
				for j in range((index1[1]-index1[0])//n+1):
					s = index1[0]+j*n
					e = index1[0]+(j+1)*n
					if e>=index1[1]:
						e = index1[1]
					if len(data[s:e])!=0:
						train.append(data[s:e])
					if testflag==0:
						train_label.append(label[s:e])
		if index2%40==0:
			print(index2)
		# print(filepath,max([len(x) for x in train]))

	train,sequence_length = padding(train,max_len)
	if testflag==0:
		train_label,_ = padding(train_label,max_len)
		return  train,train_label,sequence_length
	else:
		return train,sequence_length


def data_preprocessing1(filepaths,testflag,max_len =None):
	train = []
	if testflag ==0:
		train_label = []		
	for index2,filepath in enumerate(filepaths):
		if testflag ==0:
			labelpath = filepath.split("/")
			labelpath = os.path.join(labelpath[0],labelpath[1].split(".")[0]+".ann")
		with open(filepath,'r',encoding = "utf-8") as fr:
			file = fr.readlines()
		file = "".join(file)
		data = getfileid(file)
		if testflag==0:
			label = getlabel(labelpath,len(file))			
		if index2%40==0:
			print(index2)
		train.append(data)
		train_label.append(label)
	if testflag==0:
		max_len = max([len(x) for x in train])
	train,sequence_length = padding(train,max_len)
	if testflag==0:
		train_label,_ = padding(train_label,max_len)
		return  train,train_label,sequence_length,max_len
	else:
		return train,sequence_length

def lstm_crf():
	word2id = Word2id()	
	embedding_size = 100
	hidden_dim = 300
	tag,tag_num = tag2label()
	epochs = 10
	wordsids = tf.placeholder(tf.int32,shape = (None,None))
	labels = tf.placeholder(tf.int32,shape = (None,None))
	sequence_length = tf.placeholder(tf.int32,shape = (None))
	_word_embedding = tf.Variable(tf.truncated_normal(shape = (len(word2id),embedding_size)),dtype = tf.float32)
	word_embedding = tf.nn.embedding_lookup(_word_embedding,ids = wordsids)
	word_embedding = tf.nn.dropout(word_embedding,0.6)

	cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim)
	cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim)
	(output_fw_seq,output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
										inputs = word_embedding,
										sequence_length =sequence_length,
										dtype = tf.float32)
	outputs = tf.concat([output_fw_seq,output_bw_seq],axis= -1)
	outputs = tf.nn.dropout(outputs,0.6)
	s = tf.shape(outputs)
	w = tf.get_variable("w",shape =[2*hidden_dim,len(tag)],dtype = tf.float32,name = "w")
	b = tf.get_variable("b",shape = [len(tag)],dtype = tf.float32,name = "b")
	outputs = tf.reshape(outputs,[-1,2*hidden_dim])
	pred = tf.matmul(outputs,w)+b
	logits = tf.reshape(pred,(-1,s[1],len(tag)))

	log_likelihood ,transition_params = tf.contrib.crf.crf_log_likelihood(inputs = logits,
										tag_indices = labels,
										sequence_lengths =sequence_length)
	loss = -tf.reduce_mean(log_likelihood)

	train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in raneg(epochs):
		for train,train_label,seq_len_list in data_preprocessing():
			feed_dict = {wordsids:train,labels:train_label,sequence_length:sequence_length}
			sess.run(train_op,feed_dict = feed_dict )
		losses = sess.run(loss,feed_dict = feed_dict)
		print(epoch,losses)


def get_ann(val_pred,file):
	classes = ["O","Disease","Reason","Symptom","Test","Test_Value","Drug","Frequency","Amount","Method","Treatment","Operation","SideEff","Anatomy","Level","Duration"]	
	tag,tag_num = tag2label()
	classes_position  = {}
	for c in classes:
		classes_position[c] = []
	flag_end = 0
	flag_start = 1
	# for i in range(len(file)):
	# 	if labels[i]==0:
	# 		if flag_end ==0:
	# 			continue
	# 		else:
	# 			classes_position[classes1].append([position_s,position_s+l])
	# 			flag_end = 0
	# 			continue
	# 	se_class = tag_num[labels[i]].split("-")
	# 	if se_class[0] =="B":
	# 		if flag_end ==1:
	# 			classes_position[classes1].append([position_s,position_s+l])
	# 			flag_end =0		
	# 		classes1 = se_class[1]
	# 		position_s = i
	# 		l = 1
	# 		flag_end = 1
	# 	else:
	# 		l+=1
	for i in range(len(file)):
		if val_pred[i]==0:
			if flag_end ==0:
				continue
			else:
				classes_position[classes1].append([position_s,position_s+l])
				flag_end =0
				continue
		se_class = tag_num[val_pred[i]].split("-")
		if flag_start ==1:
			se_class1 = se_class
			position_s = i
			l = 0
			flag_start = 0
			flag_end = 1
			classes1 = se_class[1]
		if se_class1[1]!=se_class[1]:
			classes_position[classes1].append([position_s,position_s+l])
			se_class1 = se_class
			classes1 = se_class[1]
			position_s = i
			l = 0
			flag_end = 1			
		l+=1




	ann = ""
	index = 1
	for c in classes_position:
		for position in classes_position[c]:
			n = [x.start() for x in re.finditer("\n",file[position[0]:position[1]])]
			if len(n)==0:
				ann+="T"+str(index)+"\t"+c+" "+" ".join([str(x) for x in position])+"\t"+file[position[0]:position[1]]+"\n"
			else:
				position = list(set(position+n))
				position[0] = position[0]-1
				p = ""
				for i in range(len(position)-1):
					if position[i]+1<position[i+1]:

						p+=str(position[i]+1)+" "+str(position[i+1])+";"
				p = p[:-1]
				f = file[position[0]:position[1]]
				f = f.replace("\n"," ")
				ann+="T"+str(index)+"\t"+c+" "+p+"\t"+f+"\n"
			index+=1
	return ann



def get_ann1(val_pred,file):
	classes = ["O","Disease","Reason","Symptom","Test","Test_Value","Drug","Frequency","Amount","Method","Treatment","Operation","SideEff","Anatomy","Level","Duration"]	
	tag,tag_num = tag2label()
	classes_position  = {}
	for c in classes:
		classes_position[c] = []
	flag_end = 0
	flag_start = 1

	for i in range(len(file)):
		if val_pred[i]==0:
			continue
		se_class = tag_num[val_pred[i]].split("-")[1]
		if flag_start ==1:
			se_class1 = se_class
			flag_start = 0
			start = i
		if se_class!=se_class1:
			classes_position[c].append([start,i])
			se_class1 = se_class
			start = i		
	ann = ""
	index = 1
	for c in classes_position:
		for position in classes_position[c]:
				ann+="T"+str(index)+"\t"+c+" "+" ".join([str(x) for x in position])+"\t"+file[position[0]:position[1]].replace("\n"," ")+"\n"


	return ann


if __name__=="__main__":
	# lstm_crf()
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	start = time.time()
	testflag = 0
	word2id = Word2id()	
	embedding_size = 100
	hidden_dim = 100
	tag,tag_num = tag2label()
	epochs = 10
	wordsids = tf.placeholder(tf.int32,shape = (None,None))
	labels = tf.placeholder(tf.int32,shape = (None,None))
	sequence_length = tf.placeholder(tf.int32,shape = (None))
	_word_embedding = tf.get_variable('word_embedding',shape = (len(word2id),embedding_size),dtype = tf.float32)
	word_embedding = tf.nn.embedding_lookup(_word_embedding,ids = wordsids)
	word_embedding = tf.nn.dropout(word_embedding,0.6)

	cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim)
	cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim)
	(output_fw_seq,output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
										inputs = word_embedding,
										sequence_length =sequence_length,
										dtype = tf.float32)
	outputs = tf.concat([output_fw_seq,output_bw_seq],axis= -1)
	outputs = tf.nn.dropout(outputs,0.6)
	s = tf.shape(outputs)
	w = tf.get_variable("w",shape =[2*hidden_dim,len(tag)],dtype = tf.float32)
	b = tf.get_variable("b",shape = [len(tag)],dtype = tf.float32)
	outputs = tf.reshape(outputs,[-1,2*hidden_dim])
	# pred = tf.nn.relu(tf.matmul(outputs,w)+b)
	pred  = tf.matmul(outputs,w)+b
	logits = tf.reshape(pred,(-1,s[1],len(tag)))
	# logits = tf.nn.softmax(logits)

	log_likelihood ,transition_params = tf.contrib.crf.crf_log_likelihood(inputs = logits,
										tag_indices = labels,
										sequence_lengths =sequence_length)
	loss = tf.reduce_mean(-log_likelihood)
	# with tf.device("/gpu:3"):
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	sess.run(tf.global_variables_initializer())

	batch_size = 100
	if testflag ==1:
		filepaths = glob("ruijin_round1_train2_20181022/*.txt")
		print("#########length of filepaths:",len(filepaths))
		train,train_label,seq_len_list = data_preprocessing(filepaths,0)

		print("length of train:%d,max seq_len:%d"%(len(train),max(seq_len_list)))
		for epoch in range(epochs):

			for i in range(len(train)//batch_size):
				feed_dict = {wordsids:train[i*batch_size:(i+1)*batch_size],labels:train_label[i*batch_size:(i+1)*batch_size],sequence_length:seq_len_list[i*batch_size:(i+1)*batch_size]}
				sess.run(train_op,feed_dict = feed_dict )
				losses = sess.run(loss,feed_dict = feed_dict)
				print(epoch,i,losses)	
				if losses!=losses:
					break
			if losses!=losses:
				break


		testpaths = glob("ruijin_round1_test_a_20181022/*.txt")
		for testpath in testpaths:
			test,seq_len_list_test = data_preprocessing([testpath],testflag)
			feed_dict_test = {wordsids:test,sequence_length:seq_len_list_test}
			val_logits,t_p = sess.run([logits,transition_params],feed_dict_test)
			val_pred = []
			for i in range(len(val_logits)):
				 pred_labels,viterbi_score = tf.contrib.crf.viterbi_decode(val_logits[i][:seq_len_list_test[i]],t_p)
				 val_pred.extend(pred_labels)
			with open (testpath,'r',encoding = "utf-8") as fr:
				file = fr.readlines()
			file = "".join(file)
			ann = get_ann1(val_pred,file)
			submitpath = "submit/"+testpath.split("/")[1][:-3]+"ann"
			with open(submitpath,'w') as fr:
				fr.write(ann)



	else:
		filepaths = np.array(glob("ruijin_round1_train2_20181022/*.txt"))
		index = np.arange(len(filepaths))
		random.shuffle(index)
		train_index = index[:len(index)*9//10]
		test_index = index[len(train_index):]
		train,train_label,seq_len_list = data_preprocessing(filepaths[train_index],testflag)
		val,val_label,seq_len_list_val = data_preprocessing(filepaths[test_index],testflag)

		feed_dict_val = {wordsids:val,sequence_length:seq_len_list_val}

		print("length of train:%d,max seq_len:%d"%(len(train),max(seq_len_list)))
		for epoch in range(epochs):
			for i in range(len(train)//batch_size):
				feed_dict = {wordsids:train[i*batch_size:(i+1)*batch_size],labels:train_label[i*batch_size:(i+1)*batch_size],sequence_length:seq_len_list[i*batch_size:(i+1)*batch_size]}
				sess.run(train_op,feed_dict = feed_dict )
				losses = sess.run(loss,feed_dict = feed_dict)
				print(epoch,i,losses)	
			print('####################',epoch,sess.run(losses,feed_dict_val))

		val_logits,t_p = sess.run([logits,transition_params],feed_dict_val)
		val_pred =[]
		val_label1 = []
		for i in range(len(val_logits)):
			pred_labels,viterbi_score = tf.contrib.crf.viterbi_decode(val_logits[i][:seq_len_list_val[i]],t_p)
			val_label1.extend(val_label[i][:seq_len_list_val[i]])
			val_pred.extend(pred_labels)
		with open('logs/logs_{}'.format(datetime.datetime.now().strftime("%m-%d %H:%M")),'w') as fr:
			fr.write(classification_report(val_label1,val_pred)) 



	print(time.time()-start)



