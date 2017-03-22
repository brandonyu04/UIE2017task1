import pickle
import numpy as np
import sklearn
import time
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import cv2
import csv
t = time.time()
def read_prepocess_data(data_path):
	data1 = pickle.load(open(data_path, 'rb'))
	seg_pic = np.multiply(data1['rgb'],data1['segmentation']>100)
	#seg_pic1 = np.divide(np.multiply(data1['segmentation'].astype(float),data1['depth'].astype(float)),255)
	#seg_pic_depth1 = seg_pic1
	#seg_pic_depth1[:,:,:,0] = np.multiply(seg_pic1[:,:,:,0],data1['depth'])
	#seg_pic_depth1[:,:,:,1]= np.multiply(seg_pic1[:,:,:,1],data1['depth'])
	#seg_pic_depth1[:,:,:,2] = np.multiply(seg_pic1[:,:,:,2],data1['depth'])
	#seg_pic1 = np.resize(seg_pic1,(3800,120*90))
	#seg_pic_depth1_vec = np.concatenate((seg_pic1, np.reshape(data1['subjectLabels'], (-1, 1))), axis=1)
	labels=data1['gestureLabels']
	return (seg_pic,labels)
def compute_HOG(seg_pic):
	winSize = (64,64)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	feat_vec = np.asarray(hog.compute(seg_pic[0,:,:])).T
	for i in range(seg_pic.shape[0]-1):
		h = hog.compute(seg_pic[i+1,:,:])
		feat_vec = np.append(feat_vec,np.asarray(h).T,axis=0)
		if np.remainder(i, 100)==98:
			print('finish '+str(i+2)+' HOG time='+str((time.time()-t)/60))
	return feat_vec

path1 = './a1_dataTrain_chunks/a1_dataTrain_1.pkl'
path2 = './a1_dataTrain_chunks/a1_dataTrain_2.pkl'
path3 = './a1_dataTrain_chunks/a1_dataTrain_3.pkl'
path4 = './a1_dataTrain_chunks/a1_dataTrain_4.pkl'
path5 = './a1_dataTrain_chunks/a1_dataTrain_5.pkl'
path6 = './a1_dataTrain_chunks/a1_dataTrain_6.pkl'
path7 = './a1_dataTrain_chunks/a1_dataTrain_7.pkl'
path8 = './a1_dataTrain_chunks/a1_dataTrain_8.pkl'
path9 = './a1_dataTrain_chunks/a1_dataTrain_9.pkl'
path10 = './a1_dataTrain_chunks/a1_dataTrain_10.pkl'

seg_pic1,labels1 = read_prepocess_data(path1)
seg_pic2,labels2 = read_prepocess_data(path2)
seg_pic3,labels3 = read_prepocess_data(path3)
seg_pic4,labels4 = read_prepocess_data(path4)
seg_pic5,labels5 = read_prepocess_data(path5)
seg_pic6,labels6 = read_prepocess_data(path6)
seg_pic7,labels7 = read_prepocess_data(path7)
seg_pic8,labels8 = read_prepocess_data(path8)
seg_pic9,labels9 = read_prepocess_data(path9)
seg_pic10,labels10 = read_prepocess_data(path10)

print('finish reading data, start HOG time='+str((time.time()-t)/60))

feat_vec1 = compute_HOG(seg_pic1)
print('finish 1 trunk time='+str((time.time()-t)/60))
feat_vec2 = compute_HOG(seg_pic2)
print('finish 2 trunk time='+str((time.time()-t)/60))
feat_vec3 = compute_HOG(seg_pic3)
print('finish 3 trunk time='+str((time.time()-t)/60))
feat_vec4 = compute_HOG(seg_pic4)
print('finish 4 trunk time='+str((time.time()-t)/60))
feat_vec5 = compute_HOG(seg_pic5)
print('finish 5 trunk time='+str((time.time()-t)/60))
feat_vec6 = compute_HOG(seg_pic6)
print('finish 6 trunk time='+str((time.time()-t)/60))
feat_vec7 = compute_HOG(seg_pic7)
print('finish 7 trunk time='+str((time.time()-t)/60))
feat_vec8 = compute_HOG(seg_pic8)
print('finish 8 trunk time='+str((time.time()-t)/60))
feat_vec9 = compute_HOG(seg_pic9)
print('finish 6 trunk time='+str((time.time()-t)/60))
feat_vec10 = compute_HOG(seg_pic10)
print('finish 10 trunk time='+str((time.time()-t)/60))
feat_vec_train = np.concatenate((feat_vec1,feat_vec2,feat_vec3,feat_vec4,feat_vec5, feat_vec6,feat_vec7,feat_vec8,feat_vec9,feat_vec10),axis=0)

labels_train = np.concatenate((labels1,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9,labels10),axis=0)
np.save('trainHOGonTrunks',feat_vec_train)
print('finish save HOG on Trunks time='+str((time.time()-t)/60))

#feat_vec_train = np.load('trainHOGonTrunks.npy')
print('Successfully load HOG, start to train Linear SVC, time='+str(time.time()))

clf=LinearSVC(verbose=True)

print('finish HOG, start to train Linear SVC, time='+str((time.time()-t)/60))
clf.fit(feat_vec_train,labels_train)
print('Finish train Linear SVC!!!!!!!!!!, time='+str((time.time()-t)/60))
joblib.dump(clf, 'clf_on_trunk.pkl') 
print('Finish store SVC, time='+str((time.time()-t)/60))
test_path = './a1_dataTest.pkl'
test_data = pickle.load(open(test_path, 'rb'))
test_images = np.multiply(test_data['rgb'],test_data['segmentation']>100)
print('Linear: Begin to compute test image HOG, time='+str((time.time()-t)/60))
test_feat = compute_HOG(test_images)
np.save('testHOG',test_feat)
print('Finish test HOG, predict on test set, time='+str((time.time()-t)/60))
predict = clf.predict(test_feat)
print ('Finish predicting!!!!, writing CSV')
with open('TrunkLinear_results_on_HOG.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(predict.shape[0]):
        writer.writerow({'Id': str(i+1), 'Prediction': str(predict[i])})
print ('Finish!!!!')









