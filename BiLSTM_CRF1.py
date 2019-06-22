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
import prepro_data
from tqdm import tqdm
from datetime import datetime
import prepro_data1
import matplotlib.pyplot as plt
import my_metrics

def testdata_prepro(testpath):
	line_len = 20
	span = 100
	flag_char = ['。','？',',',';','!','.']		

	with open("data/wordsids_classes1.pkl",'rb') as fr:
		wordsids,classes = pickle.load(fr)
	split_char = [wordsids['。'],wordsids[';']]

	with open(testpath,'r',encoding = "utf-8") as fr:
		file = fr.readlines()
		for i in range(len(file)):
			if len(file[i])>1 and len(file[i])<line_len and file[i][-2] not in flag_char:
				file[i] = file[i][:-1]+"。"

		file = "".join(file)
	data = []
	for f in file:
		if f=="\n":
			c = wordsids['LB']
		elif f ==" " or f=="\u2003":
			c = wordsids['SPACE']
		elif f.isdigit():
			c = wordsids["NUM"]
		else:
			c = wordsids.get(f,wordsids['UNK'])
		data.append(c)
	data1 = []
	start = 0
	for i in range(len(data)):
		if data[i] in split_char or i==len(data)-1:
			if i+1-start<=span:
				data1.append(data[start:i+1])
			else:
				for j in range((i+1-start)//span+1):
					s = start+span*j
					e = start+(j+1)*span
					if e>i+1:
						e=i+1
					data1.append(data[s:e])
			start = i+1
	data1 = [x for x in data1 if len(x)!=0]
	seq_len_list = [len(x) for x in data1]
	for i in range(len(data1)):
		data1[i] = data1[i]+[wordsids['PAD']]*(span-len(data1[i]))
	data1 = np.array(data1)
	return data1,file,seq_len_list
def createann(val_pred,file):
	with open("data/wordsids_classes.pkl",'rb') as fr:
		_,classes = pickle.load(fr)
	tags= []
	val_pred = [classes[x] for x in val_pred]
	for i in range(len(val_pred)):
		if val_pred[i]!="O":
			tag = val_pred[i].split("-")[1]
		else:
			tag = "O"
		tags.append(tag)
	pre = tags[0]
	start = 0
	ann = ""
	index = 1
	for i in range(1,len(tags)):
		cur = tags[i]

		if cur!=pre:
			if pre!="O":
				c = file[start:i].replace("\n"," ")
				ann+="T"+str(index)+"\t"+pre+" "+str(start)+" "+str(i)+"\t"+c+"\n"
				index+=1
			start = i
			pre = cur
	return ann






if __name__=="__main__":
	# lstm_crf()
	nowtime = datetime.now().strftime("%d_%H_%M")
	train,val,train_label,val_label,seq_len_list,seq_len_list_val = prepro_data1.get_trainval_tf(0.1,False)
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	start1 = time.time()
	testflag = 0
	with open("data/wordsids_classes1.pkl",'rb') as fr:
		word2id,classes = pickle.load(fr)
	embedding_size = 100
	hidden_dim = 100
	epochs = 50
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
	w = tf.get_variable("w",shape =[2*hidden_dim,len(classes)],dtype = tf.float32)
	b = tf.get_variable("b",shape = [len(classes)],dtype = tf.float32)
	outputs = tf.reshape(outputs,[-1,2*hidden_dim])
	# pred = tf.nn.relu(tf.matmul(outputs,w)+b)
	pred  = tf.matmul(outputs,w)+b
	logits = tf.reshape(pred,(-1,s[1],len(classes)))
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
	all_loss = []
	if testflag ==1:
		filepaths = glob("ruijin_round1_train2_20181022/*.txt")
		print("#########length of filepaths:",len(filepaths))
		print("length of train:%d,max seq_len:%d"%(len(train),max(seq_len_list)))
		
		train = np.concatenate((train,val))
		train_label = np.concatenate((train_label,val_label))
		seq_len_list = np.concatenate((seq_len_list,seq_len_list_val))
		for epoch in range(epochs):
			epoch_loss = []
			for i in range(len(train)//batch_size):
				feed_dict = {wordsids:train[i*batch_size:(i+1)*batch_size],labels:train_label[i*batch_size:(i+1)*batch_size],sequence_length:seq_len_list[i*batch_size:(i+1)*batch_size]}
				sess.run(train_op,feed_dict = feed_dict )
				losses = sess.run(loss,feed_dict = feed_dict)
				epoch_loss.append(losses)
				if i%100==0:
					print(epoch,i,losses)
			all_loss.append(np.mean(epoch_loss))	



		testpaths = glob("data/ruijin_round1_test_a_20181022/*.txt")
		savepath = "submit/submit_{}".format(nowtime)
		os.makedirs(savepath)
		for testpath in tqdm(testpaths):
			test,file,seq_len_list_test = testdata_prepro(testpath)			

			feed_dict_test = {wordsids:test,sequence_length:np.array(seq_len_list_test)}
			val_logits,t_p = sess.run([logits,transition_params],feed_dict_test)
			val_pred = []
			for i in range(len(val_logits)):
				 pred_labels,viterbi_score = tf.contrib.crf.viterbi_decode(val_logits[i][:seq_len_list_test[i]],t_p)
				 val_pred.extend(pred_labels)
			with open (testpath,'r',encoding = "utf-8") as fr:
				file = fr.readlines()
			file = "".join(file)
			ann = createann(val_pred,file)
			submitpath = savepath+"/"+testpath.split("/")[-1][:-3]+"ann"
			with open(submitpath,'w') as fr:
				fr.write(ann)

		saver = tf.train.Saver()
		saver.save(sess,"model/model_"+nowtime)
		plt.plot(all_loss,'r')
		plt.show()

	else:
		feed_dict_val = {wordsids:val,sequence_length:seq_len_list_val,labels:val_label}
		val_loss = []
		print("length of train:%d,val:%d,max seq_len:%d"%(len(train),len(val),seq_len_list.max()))
		for epoch in range(epochs):
			start = time.time()
			epoch_loss = []
			for i in range(len(train)//batch_size+1):
				feed_dict = {wordsids:train[i*batch_size:(i+1)*batch_size],labels:train_label[i*batch_size:(i+1)*batch_size],sequence_length:seq_len_list[i*batch_size:(i+1)*batch_size]}
				sess.run(train_op,feed_dict = feed_dict )
				losses = sess.run(loss,feed_dict = feed_dict)
				epoch_loss.append(losses)
				if i%100==0:
					print(epoch,i,losses)	
			temp = sess.run(loss,feed_dict_val)
			print('#################### epoch:{},loss:{},val_loss:{},cost_time:{}'.format(epoch,np.mean(epoch_loss),temp,time.time()-start))
			all_loss.append(np.mean(epoch_loss))
			val_loss.append(temp)
		val_logits,t_p = sess.run([logits,transition_params],feed_dict_val)
		val_pred =[]
		val_label1 = []
		for i in range(len(val_logits)):
			pred_labels,viterbi_score = tf.contrib.crf.viterbi_decode(val_logits[i][:seq_len_list_val[i]],t_p)
			val_label1.extend(val_label[i][:seq_len_list_val[i]])
			val_pred.extend(pred_labels)
		with open('logs/logs_{}.txt'.format(nowtime),'w') as fr:
			fr.write(classification_report(val_label1,val_pred,target_names = classes)) 

		idswords = dict(zip(word2id.values(),word2id.keys()))
		file = ""
		for s in val:
			for w in s:
				w_id = idswords.get(w,"UNK")
				if w_id=="SPACE":
					d = " "
				elif w_id =="LB":
					d ="\n"
				elif w_id =="NUM":
					d = "1"
				elif w_id =="UNK":
					d = "K"
				else:
					d = w_id
				if d!="PAD":
					file +=d

		ann = createann(val_pred,file)
		predpath = "measure/pred.ann"
		with open(submitpath,'w') as fr:
			fr.write(ann)	

		ann = createann(val_label1,file)
		truepath = "measure/true.ann"
		with open(submitpath,'w') as fr:
			fr.write(ann)

		my_metrics.my_f1(truepath,predpath)
		saver = tf.train.Saver()
		saver.save(sess,"model/model_"+nowtime)
		plt.plot(all_loss,'r')
		plt.plot(val_loss,'b')
		plt.show()



	print(time.time()-start1)



