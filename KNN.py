#coding:utf-8
from numpy import *
import operator
import os

'''
#包含了样本集
def creatDataSet():
	#numpy函数中的array创建一个4*2的矩阵
	group = array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
	#标记分类
	labels = ['A','A','B','B']
	return group,labels
'''


#KNN算法进行分类
def KNNClassify(newInput,dataSet,labels,k):
	#shape[0]返回矩阵的行数，这里即为样本数量。
	numSamples = dataSet.shape[0]
	#计算欧氏距离
	diff = tile(newInput,reps = [numSamples,1]) - dataSet #把新加数据复制，和样本矩阵相同的维度，做差值。
	squaredDiff = diff ** 2
	squaredDist = sum(squaredDiff, axis = 1)
	distance = squaredDist ** 0.5
	#给距离排序，返回的是原来数字的下标值，即排在当前位置的数字是原来list中的第几个数字。
	sortedDistIndex = argsort(distance)

	classCount = {}
	for i in xrange(k):
		voteLabel = labels[sortedDistIndex[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1

	maxCount = 0
	for key,value in classCount.items():
		if value > maxCount:
			maxCount = value
			maxIndex = key

	return maxIndex

#把每个文件的内容转化成向量
def toVector(filename):
	rows = 32
	cols = 32
	tVector = zeros((1,rows*cols))
	file1 = open(filename)
	for row in xrange(rows):
		lines = file1.readline()
		for col in xrange(cols):
			tVector[0,row*32 + col] = int(lines[col])

	return tVector

#传数据集
def loadDataSet():
	print "---Getting training set..."
	dataSetDir = 'D:/Python/workspace/KNN_data/'
	trainingFileList = os.listdir(dataSetDir + 'trainingDigits')
	numSamples = len(trainingFileList)

	train_x = zeros((numSamples,1024))
	train_y = []
	for i in xrange(numSamples):
		filename = trainingFileList[i]

		train_x[i] = toVector(dataSetDir + 'trainingDigits/%s' % filename)
		label = int(filename.split('_')[0])
		train_y.append(label)

	print "---Getting testing set..."
	testingFileList = os.listdir(dataSetDir + 'testDigits')
	numSamples = len(testingFileList)

	test_x = zeros((numSamples,1024))
	test_y = []
	for i in xrange(numSamples):
		filename = testingFileList[i]

		test_x[i] = toVector(dataSetDir + 'testDigits/%s' % filename)
		label = int(filename.split('_')[0])
		test_y.append(label)

	return train_x,train_y,test_x,test_y

def testHandWritingClass():
	print "Step 1: load data..."
	train_x,train_y,test_x,test_y = loadDataSet()

	print "Step 2: training..."
	pass

	print "Step 3: testing..."
	numTestSamples = test_x.shape[0]
	matchCount = 0
	for i in xrange(numTestSamples):
		predict = KNNClassify(test_x[i], train_x, train_y, 3)
		if predict == test_y[i]:
			matchCount += 1

	accuracy = float(matchCount) / numTestSamples

	print "Step 4: show the result..."
	print "The classify accuracy is: %.2f%%" % (accuracy * 100)

