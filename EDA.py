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








