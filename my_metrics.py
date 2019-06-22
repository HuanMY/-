import pandas as pd


def my_f1(truepath,predpath):
	classes = ["Disease","Reason","Symptom","Test","Test_Value","Drug","Frequency","Amount","Method","Treatment","Operation","SideEff","Anatomy","Level","Duration"]
	true = pd.read_table(truepath,header=None,names = ["id","classes_position","words"])
	pred = pd.read_table(predpath,header=None,names = ["id","classes_position","words"])
	cla_pos_true = {}
	cla_pos_pred = {}
	for c in classes:
		cla_pos_true[c] = []
	for c in classes:
		cla_pos_pred[c] = []
	for index,row in true.iterrows():
		c_p = row['classes_position'].split(" ")
		c = c_p[0]
		p_s,p_e = int(c_p[1]),int(c_p[-1])
		cla_pos_true[c].append([p_s,p_e])
	for index,row in pred.iterrows():
		c_p = row['classes_position'].split(" ")
		c = c_p[0]
		p_s,p_e = int(c_p[1]),int(c_p[-1])
		cla_pos_pred[c].append([p_s,p_e])
	P1 = 0
	P2 = 0
	for c in cla_pos_pred:
		if len(cla_pos_pred[c])!=0:
			for p in cla_pos_pred[c]:
				P2 +=p[1]-p[0]
				for p1 in cla_pos_true[c]:
					if p[0]>p1[1] or p[1]<p1[0]:
						continue
					else:
						P1+=min([p1[1],p[1]])-max([p1[0],p[0]])
	P = P1/P2
	R1 = 0
	R2 = 0
	for c in cla_pos_pred:
		if len(cla_pos_true[c])!=0:
			for p in cla_pos_true[c]:
				R2+=p[1]-p[0]
				for p1 in cla_pos_pred[c]:
					if p[0]>p1[1] or p[1]<p1[0]:
						continue
					else:
						R1+=min([p1[1],p[1]])-max([p1[0],p[0]])
	R = R1/R2
	f1 = 2*P*R/(P+R)
	return f1
if __name__ =="__main__":
	f1 = my_f1("data/ruijin_round1_train2_20181022/0.ann","data/ruijin_round1_train2_20181022/0.ann")

