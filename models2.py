import keras
import pandas as pd
import numpy as np
import os
from glob import glob
import prepro_data
from keras.models import Sequential,Model
from keras.layers import Layer,Embedding,Bidirectional,LSTM,Dense,TimeDistributed,Dropout,merge,Input,Conv1D,ZeroPadding1D,concatenate
from keras.callbacks import ModelCheckpoint
from keras_contrib.layers import CRF
import time
from sklearn.metrics import classification_report
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import pickle
import prepro_data2
import my_metrics
import keras.backend as K
from keras.layers.core import *

now = datetime.now()
nowtime = now.strftime("%d_%H_%M")

class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		super(AttentionLayer, self).__init__(** kwargs)
	
	def build(self, input_shape):
		assert len(input_shape)==3
		# W.shape = (time_steps, time_steps)
		self.W = self.add_weight(name='att_weight', 
								 shape=(input_shape[1], input_shape[1]),
								 initializer='uniform',
								 trainable=True)
		super(AttentionLayer, self).build(input_shape)

	def call(self, inputs, mask=None):
		print(inputs.shape)
		input_dim = int(inputs.shape[2])
		# TIME_STEPS = int(inputs.shape[1])
		a = Permute((2, 1))(inputs)
		a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
		a = Dense(TIME_STEPS, activation='softmax')(a)
		# if SINGLE_ATTENTION_VECTOR:
		#     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
		#     a = RepeatVector(input_dim)(a)
		a_probs = Permute((2, 1), name='attention_vec')(a)
		output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])
		return output_attention_mul

	def compute_output_shape(self, input_shape):
		return input_shape[0],input_shape[1],input_shape[2]

def attention_3d_block(inputs):
	TIME_STEPS = 100
	input_dim = int(inputs.shape[2])
	a = Permute((2, 1))(inputs)
	a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
	a = Dense(TIME_STEPS, activation='softmax')(a)
	a_probs = Permute((2, 1), name='attention_vec')(a)
	output_attention_mul = merge.multiply([inputs, a_probs], name='attention_mul')
	return output_attention_mul


class My_Model:
	def __init__(self,batch_size,epochs):
		self.batch_size = batch_size
		self.epochs = epochs
	def gpu_config(self):
		os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		set_session(tf.Session(config = config))

	def BiLSTM_CRF(self,wordsids,classes,train,val,train_label,val_label,istrain = True):
		# now = datetime.now()
		span = 400
		self.gpu_config()
		output_dim = span
		lstm_cell = span
		model = Sequential()
		model.add(Embedding(len(wordsids),output_dim, mask_zero=True))
		model.add(Bidirectional(LSTM(lstm_cell,return_sequences = True,dropout_W = 0.1,dropout_U = 0.1)))
		model.add(Dropout(0.6))
		# model.add(TimeDistributed(Dense(len(classes))))
		crf = CRF(len(classes),sparse_target = True)
		model.add(crf)
		model.summary()

		checkpoint = ModelCheckpoint("model/model_{}.h5".format(now.strftime("%d_%H_%M")),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		model.compile(keras.optimizers.Adam(1e-2),loss = crf.loss_function,metrics = [crf.accuracy])
		if istrain:

			history = model.fit(train,
					train_label,
					self.batch_size,
					epochs = self.epochs,
					validation_data = (val,val_label),
					# steps_per_epoch = 1,
					# validation_steps = 1,
					callbacks = [checkpoint]
					)

			return model,history
		else:
			return model

	def BiLSTM_CRF_attention(self,wordsids,classes,train,val,train_label,val_label,istrain = True):
		now = datetime.now()
		self.gpu_config()
		output_dim = 50
		lstm_cell = 50
		span = 100
		inputs = Input(shape = (span,))	
		emb = Embedding(len(wordsids),output_dim,mask_zero =False)(inputs)
		bd = Bidirectional(LSTM(lstm_cell,return_sequences = True))(emb)
		bd_d = Dropout(0.5)(bd)
		td = TimeDistributed(Dense(len(classes)))(bd_d)
		at = attention_3d_block(td)
		crf = CRF(len(classes),sparse_target = True)
		model = Model(inputs,crf(at))
		checkpoint = ModelCheckpoint("model/model_{}.h5".format(now.strftime("%d_%H_%M")),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		model.compile(keras.optimizers.Adam(1e-2),loss = crf.loss_function,metrics = [crf.accuracy])
		if istrain:

			history = model.fit(train,
					train_label,
					self.batch_size,
					epochs = self.epochs,
					validation_data = (val,val_label),
					# steps_per_epoch = 1,
					# validation_steps = 1,
					callbacks = [checkpoint]
					)

			return model,history
		else:
			return model



	def BiLSTM_CRF_test(self,wordsids,classes,train,train_label,istrain = True):
		now = datetime.now()
		self.gpu_config()
		span = 200
		output_dim = span
		lstm_cell = span
		model = Sequential()
		model.add(Embedding(len(wordsids),output_dim, mask_zero=True))
		model.add(Bidirectional(LSTM(lstm_cell,return_sequences = True,dropout_W = 0.1,dropout_U = 0.1)))
		model.add(Dropout(0.4))
		# model.add(TimeDistributed(Dense(len(classes))))
		crf = CRF(len(classes),sparse_target = True)
		model.add(crf)
		model.summary()

		# checkpoint = ModelCheckpoint("model/model_{}.h5".format(now.strftime("%d_%H_%M")),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		model.compile(keras.optimizers.Adam(1e-2),loss = crf.loss_function,metrics = [crf.accuracy])
		if istrain:

			history = model.fit(train,
					train_label,
					self.batch_size,
					epochs = self.epochs,
					# validation_data = (val,val_label),
					# steps_per_epoch = 1,
					# validation_steps = 1,
					# callbacks = [checkpoint]
					)

			return model,history
		else:
			return model
	def CNN_CRF(self,wordsids,classes,train,val,train_label,val_label,istrain =True):
		self.gpu_config()
		output_dim = 50
		lstm_cell = 50
		max_len = 100
		model = Model()
		inputs = Input(shape = (None,))
		word_emd = Embedding(len(wordsids),output_dim)(inputs)
		paddinglayer = ZeroPadding1D(2)(word_emd)
		conv1 = Conv1D(32,2*2+1,border_mode = "valid")(paddinglayer)
		conv1_d = Dropout(0.1)(conv1)
		conv_dense = TimeDistributed(Dense(100))(conv1_d)
		crf = CRF(len(classes),sparse_target = True)
		crf_output = crf(conv_dense)
		model = Model(inputs,crf_output)
		model.compile(optimizer = keras.optimizers.Adam(1e-2),
					loss = crf.loss_function,
					metrics = [crf.accuracy])
		model.summary()		
		checkpoint = ModelCheckpoint("model/model_{}.h5".format(nowtime),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		if istrain:
			history = model.fit(train,
								train_label,
								self.batch_size,
								epochs = self.epochs,
								callbacks = [checkpoint],
								validation_data = (val,val_label)
								)			

			return model,history
		else:
			return model


	def BiLSTM_CNN_CRF(self,wordsids,classes,train,val,train_label,val_label,istrain =True):
		self.gpu_config()
		output_dim = 50
		lstm_cell = 50
		max_len = 100
		model = Model()
		inputs = Input(shape = (None,))
		word_emd = Embedding(len(wordsids),output_dim)(inputs)
		bilstm = Bidirectional(LSTM(lstm_cell,return_sequences = True,dropout_W = 0.1,dropout_U = 0.1))(word_emd)
		bilstm_d = Dropout(0.1)(bilstm)
		paddinglayer = ZeroPadding1D(2)(word_emd)
		conv1 = Conv1D(32,2*2+1,border_mode = "valid")(paddinglayer)
		conv1_d = Dropout(0.1)(conv1)
		conv_dense = TimeDistributed(Dense(100))(conv1_d)
		rnn_cnn_merge = concatenate([bilstm_d,conv_dense],axis = 2)
		dense = TimeDistributed(Dense(lstm_cell))(rnn_cnn_merge)
		crf = CRF(len(classes),sparse_target = True)
		crf_output = crf(dense)
		model = Model(inputs,crf_output)
		model.compile(optimizer = keras.optimizers.Adam(1e-2),
					loss = crf.loss_function,
					metrics = [crf.accuracy])
		model.summary()		
		checkpoint = ModelCheckpoint("model/model_{}.h5".format(nowtime),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		if istrain:
			history = model.fit(train,
								train_label,
								self.batch_size,
								epochs = self.epochs,
								callbacks = [checkpoint],
								validation_data = (val,val_label)
								)			

			return model,history
		else:
			return model

	def BiLSTM_CNN_CRF1(self,wordsids,classes,train,val,train_label,val_label,istrain =True):
		self.gpu_config()
		output_dim = 50
		lstm_cell = 50
		max_len = 100
		model = Model()
		inputs = Input(shape = (None,))
		word_emd = Embedding(len(wordsids),output_dim)(inputs)
		bilstm = Bidirectional(LSTM(lstm_cell,return_sequences = True,dropout_W = 0.1,dropout_U = 0.1))(word_emd)
		bilstm_d = Dropout(0.3)(bilstm)
		paddinglayer = ZeroPadding1D(2)(bilstm_d)
		conv1 = Conv1D(32,2*2+1,border_mode = "valid")(paddinglayer)
		conv1_d = Dropout(0.1)(conv1)
		conv_dense = TimeDistributed(Dense(100))(conv1_d)
		crf = CRF(len(classes),sparse_target = True)
		crf_output = crf(conv_dense)
		model = Model(inputs,crf_output)
		model.compile(optimizer = keras.optimizers.Adam(1e-2),
					loss = crf.loss_function,
					metrics = [crf.accuracy])
		model.summary()		
		checkpoint = ModelCheckpoint("model/model_{}.h5".format(nowtime),monitor = "val_acc",verbose = 1,save_best_only = True,mode = "max")

		if istrain:
			history = model.fit(train,
								train_label,
								self.batch_size,
								epochs = self.epochs,
								callbacks = [checkpoint],
								validation_data = (val,val_label)
								)			

			return model,history
		else:
			return model

def measurement(val_,val_label_,model,classes,history,wordsids):
	val_pred = model.predict(val_,batch_size = 50000)[0]
	val_pred = [np.argmax(pred) for pred in val_pred]
	val_label_ = val_label_.flatten()
	c_r = classification_report(val_label_,val_pred,target_names = classes)
	now = datetime.now()
	with open("logs/logs_{}".format(now.strftime("%d_%H:%M")),'w') as fr:
		fr.write(c_r)
	print(c_r)
	idswords = dict(zip(wordsids.values(),wordsids.keys()))
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

	ann = createann(val_pred,file,classes)
	predpath = "measure/pred.ann"
	with open(predpath,'w') as fr:
		fr.write(ann)	

	ann = createann(val_label_,file,classes)
	truepath = "measure/true.ann"
	with open(truepath,'w') as fr:
		fr.write(ann)

	print("f1_score",my_metrics.my_f1(truepath,predpath))


	# plt.plot(history.history['val_loss'],'b')
	# plt.plot(history.history['loss'],'r')
	# plt.show()
	# plt.plot(history.history['val_acc'],'b')
	# plt.plot(history.history['acc'],'r')
	# plt.show()


def measurement_attention(val_1,val_label_,model,classes,history,wordsids):
	val_pred = []
	for val_2 in val_1:
		min_l = len(val_2)
		if min_l<100:
			val_2 +=[0]*(100-min_l)
		val_2 = np.array(val_2)
		val_2 = val_2.reshape((1,len(val_2)))
		val_pred1 = model.predict(val_2,batch_size = 100)[0]
		val_pred1 = [np.argmax(pred) for pred in val_pred1]
		val_pred.extend(val_pred1)
	if min_l!=100:
		val_pred = val_pred[:-(100-min_l)]
	val_pred = np.array(val_pred)
	val_label_ = val_label_.flatten()
	c_r = classification_report(val_label_,val_pred,target_names = classes)
	now = datetime.now()
	with open("logs/logs_{}".format(now.strftime("%d_%H:%M")),'w') as fr:
		fr.write(c_r)
	print(c_r)
	idswords = dict(zip(wordsids.values(),wordsids.keys()))
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

	ann = createann(val_pred,file,classes)
	predpath = "measure/pred.ann"
	with open(predpath,'w') as fr:
		fr.write(ann)	

	ann = createann(val_label_,file,classes)
	truepath = "measure/true.ann"
	with open(truepath,'w') as fr:
		fr.write(ann)

	print("f1_score",my_metrics.my_f1(truepath,predpath))


	plt.plot(history.history['val_loss'],'b')
	plt.plot(history.history['loss'],'r')
	plt.show()
	plt.plot(history.history['val_acc'],'b')
	plt.plot(history.history['acc'],'r')
	plt.show()
		


def testdata_prepro(testpath,wordsids):

	with open(testpath,'r',encoding = "utf-8") as fr:
		file = fr.readlines()
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
	data = np.array(data).reshape((1,len(data)))
	return data,file

def createann(data_pred,file,classes):

	data_pred = [classes[x] for x in data_pred]
	data_pred1 = []
	for d in data_pred:
		if d!="O":
			c = d.split("-")[1]
		else:
			c = d
		data_pred1.append(c)

	pre = data_pred1[0]
	start = 0
	ann = ""
	index = 1
	for i in range(1,len(data_pred1)):
		cur = data_pred1[i]
		if cur!=pre:
			if pre!="O":
				c =file[start:i].replace("\n",' ')
				ann+="T"+str(index) +"\t"+pre+" "+str(start)+" "+str(i)+"\t"+c+"\n"
				index+=1
			start = i
			pre = cur
	return ann


				



def test(testpath,model,wordsids,classes):
	testpaths = os.listdir(testpath)
	now = datetime.now()
	basepath = "submit/submit_{}".format(now.strftime("%d_%H_%M"))+"/"
	os.makedirs(basepath)
	for tp in tqdm(testpaths):
		data,file = testdata_prepro(os.path.join(testpath,tp),wordsids)
		data_pred = model.predict(data)[0]
		data_pred = [np.argmax(pred) for pred in data_pred]
		ann = createann(data_pred,file,classes)
		savepath = basepath+tp[:-3]+"ann"
		with open(savepath,'w',encoding = 'utf-8') as fr:
			fr.write(ann)




def test_attention(testpath,model,wordsids,classes):
	testpaths = os.listdir(testpath)
	now = datetime.now()
	basepath = "submit/submit_{}".format(now.strftime("%d_%H_%M"))+"/"
	os.makedirs(basepath)
	for tp in tqdm(testpaths):
		data,file = testdata_prepro(os.path.join(testpath,tp),wordsids)
		
		data = data.flatten()
		data_ = []
		index = [x for x in range(len(data)) if x%100==0]
		index = index+[len(data)]
		for i in range(len(index)-1):
			if index[i+1]-index[i]==0:
				continue
			data_.append(list(data[index[i]:index[i+1]]))
		data_pred = []
		for data1 in data_:
			min_l = len(data1)
			if min_l<100:
				data1+=[0]*(100-min_l)
			data1 = np.array(data1)
			data1 = data1.reshape((1,len(data1)))
			data_pred1 = model.predict(data1)[0]
			data_pred1 = [np.argmax(pred) for pred in data_pred1]
			data_pred.extend(data_pred1)
		if min_l!=100:
			data_pred = data_pred[:-(100-min_l)]
		ann = createann(data_pred,file,classes)
		savepath = basepath+tp[:-3]+"ann"
		with open(savepath,'w',encoding = 'utf-8') as fr:
			fr.write(ann)





if __name__ =="__main__":
	start = time.time()
	istrain = False
	model = My_Model(100,10)
	history = 1
	train,val,train_label,val_label,val_,val_label_,wordsids,classes = prepro_data2.get_trainval(0.1,False)

	# bilstm_model,history = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = True)
	if istrain ==False:

		bilstm_model = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = istrain)
		bilstm_model.load_weights("model/model_12_22_13.h5")
		measurement(val_,val_label_,bilstm_model,classes,history,wordsids)
		# test("data/ruijin_round1_test_b_20181112",bilstm_model,wordsids,classes)
		print(time.time()-start)


	else:
		# train = np.concatenate((train,val))
		# train_label = np.concatenate((train_label,val_label))

		# bilstm_model,history = model.BiLSTM_CRF_test(wordsids,classes,train,train_label,istrain = istrain)
		bilstm_model,history = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = istrain)

		# test("data/ruijin_round1_test_b_20181112",bilstm_model,wordsids,classes)
		# bilstm_model,history = model.BiLSTM_CRF_attention(wordsids,classes,train,val,train_label,val_label,istrain = istrain)

		print(time.time()-start)


		# val_1 = []
		# index = [x for x in range(len(val_.flatten())) if x%100==0]
		# index = index+[len(val_.flatten())]
		# for i in range(len(index)-1):
		# 	if index[i+1]-index[i]==0:
		# 		continue
		# 	val_1.append(list(val_.flatten()[index[i]:index[i+1]]))

		measurement(val_,val_label_,bilstm_model,classes,history,wordsids)		
		test("data/ruijin_round1_test_b_20181112",bilstm_model,wordsids,classes)



