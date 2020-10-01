import numpy as np
import time

class KnnClassifier:
	def __init__(self, trainingsize, traingingdata, traininglabel, testdata, k):
		self.trainingdata=traingingdata[:int(len(traingingdata)*trainingsize)]
		self.traininglabel=traininglabel[:int(len(traininglabel)*trainingsize)]
		self.testdata=testdata
		self.k=k

	def subsampling(self):		
		self.trainingdata=self.trainingdata.reshape(-1,len(self.trainingdata),28)[:,::2,::2]
		self.testdata=self.testdata.reshape(-1,len(self.testdata),28)[:,::2,::2]
		self.testdata=self.testdata[0]
		self.trainingdata=self.trainingdata[0]
		print ("Done Subsampling")

	def test(self, sb):
		tstime=0
		tctime=0
		if sb!=0:
			self.subsampling()
		self.testdata=self.testdata \
		.reshape(int(len(self.testdata)/len(self.testdata[0])),len(self.testdata[0])*len(self.testdata[0]))
		self.trainingdata=self.trainingdata \
		.reshape(int(len(self.trainingdata)/len(self.trainingdata[0])),len(self.trainingdata[0])*len(self.trainingdata[0]))
		testres=[]
		for i in range(len(self.testdata)):
			time1=time.time()
			thisp=np.tile(self.testdata[i],(len(self.trainingdata),1))
			thisp=((thisp-self.trainingdata)**2)**0.5
			thisp=thisp.T
			thisp=sum(thisp)
			time2=time.time()
			rank=np.argsort(thisp)
			firstten=[]
			for j in range(self.k):#k
				firstten.append(self.traininglabel[rank[j]])
			testres.append(np.bincount(np.array(firstten).astype(np.int64)).argmax())
			time3=time.time()
			tstime=tstime+(time3-time2)
			tctime=tctime+(time2-time1)
		print("sort time:", tctime)
		print("calculate time", tstime)
		return testres