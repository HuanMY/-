import keras
import pandas as pd
import numpy as np
import os
from glob import glob
# import prepro_data
from keras.models import Sequential,Model
from keras.layers import Embedding,Bidirectional,LSTM,Dense,TimeDistributed,Dropout,merge,Input,Conv1D,ZeroPadding1D,concatenate
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
# import prepro_data3
# import my_metrics
from keras.layers import Layer

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
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

class My_Model:
	def __init__(self,batch_size,epochs):
		self.batch_size = batch_size
		self.epochs = epochs
	def gpu_config(self):
		os.environ['CUDA_VISIBLE_DEVICES'] = '2'
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		set_session(tf.Session(config = config))

	def BiLSTM_CRF(self,wordsids,classes,train,val,train_label,val_label,istrain = True):
		now = datetime.now()
		self.gpu_config()
		output_dim = 3*60
		lstm_cell = 3*60
		model = Sequential()
		model.add(Embedding(len(wordsids),output_dim, mask_zero=True))
		model.add(Bidirectional(LSTM(lstm_cell,return_sequences = True)))
		# model.add(TimeDistributed(Dense(100)))
		# model.add(AttentionLayer())
		# model.add(TimeDistributed(Dense(100)))
		model.add(Dropout(0.4))
		model.add(TimeDistributed(Dense(len(classes))))
		# model.add(Dropout(0.5))
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
		span = 2*60
		inputs = Input(shape = (span,))	
		emb = Embedding(len(wordsinds),output_dim,mask_zero =False)(inputs)
		bd = Bidirectional(LSTM(lstm_cell,return_sequences = True))(emb)
		bd_d = Dropout(0.5)(bd)
		td = TimeDistributed(Dense(len(classes)))(bd_d)
		at = AttentionLayer()(td)
		td_d = Dropout(0.5)(at)
		crf = CRF(len(classes),sparse_target = True)(td_d)
		model = Model(inputs,crf)
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


	plt.plot(history.history['val_loss'],'b')
	plt.plot(history.history['loss'],'r')
	plt.show()
	plt.plot(history.history['val_acc'],'b')
	plt.plot(history.history['acc'],'r')
	plt.show()
		



def testdata_prepro(testpath,wordsids,classes):
	# with open("data/wordsids_classes.pkl",'rb') as fr:
	# 	wordsids,classes = pickle.load(fr)
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
	# with open("data/wordsids_classes.pkl",'rb') as fr:
	# 	wordsids,classes = pickle.load(fr)
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
		data,file = testdata_prepro(os.path.join(testpath,tp),wordsids,classes)
		data_pred = model.predict(data)[0]
		data_pred = [np.argmax(pred) for pred in data_pred]
		ann = createann(data_pred,file,classes)
		savepath = basepath+tp[:-3]+"ann"
		with open(savepath,'w',encoding = 'utf-8') as fr:
			fr.write(ann)





if __name__ =="__main__":
	start = time.time()
	istrain = True
	model = My_Model(50,10)
	with open("data/wordsids_classes.pkl",'rb') as fr:
		wordsids,classes = pickle.load(fr)
	train,val,train_label,val_label,val_,val_label_ = prepro_data.get_trainval_v1(0.1,False)

	# bilstm_model,history = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = True)
	if istrain ==False:

		bilstm_model = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = istrain)
		bilstm_model.load_weights("model/model_11_15_33.h5")
		measurement(val_,val_label_,bilstm_model,classes,history,wordsids)
		# test("data/ruijin_round1_test_a_20181022",bilstm_model)
		test("data/ruijin_round1_test_a_20181022",bilstm_model,wordsids,classes)
	
	else:
		bilstm_model,history = model.BiLSTM_CRF(wordsids,classes,train,val,train_label,val_label,istrain = True)
		print(time.time()-start)

		measurement(val_,val_label_,bilstm_model,classes,history,wordsids)


