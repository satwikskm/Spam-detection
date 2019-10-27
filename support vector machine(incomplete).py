#!/usr/bin/env python
# coding: utf-8

# # Support Vector Learning
# Used to separate the datasets
# 
# In machine learning, support-vector machines (SVMs, also support-vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.
# 
# In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
# 
# When data are unlabelled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The support-vector clustering[2] algorithm, created by Hava Siegelmann and Vladimir Vapnik, applies the statistics of support vectors, developed in the support vector machines algorithm, to categorize unlabeled data, and is one of the most widely used clustering algorithms in industrial applications
# 
# ## Application
# SVMs can be used to solve various real-world problems:
# 
# SVMs are helpful in text and hypertext categorization, as their application can significantly reduce the need for labeled training instances in both the standard inductive and transductive settings.[citation needed] Some methods for shallow semantic parsing are based on support vector machines.[6]
# 
# Classification of images can also be performed using SVMs. Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback. This is also true for image segmentation systems, including those using a modified version SVM that uses the privileged approach as suggested by Vapnik.[7][8]
# 
# Hand-written characters can be recognized using SVM.[9]
# 
# The SVM algorithm has been widely applied in the biological and other sciences. They have been used to classify proteins with up to 90% of the compounds classified correctly. Permutation tests based on SVM weights have been suggested as a mechanism for interpretation of SVM models.[10][11]
# Support-vector machine weights have also been used to interpret SVM models in the past.[12] Posthoc interpretation of support-vector machine models in order to identify features used by the model to make predictions is a relatively new area of research with special significance in the biological sciences.

# In[18]:


import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import gzip
import os


# In[19]:


Train_dir="D:\\machine_learning_and _iot\\train-mails"
Test_dir="D:\\machine_learning_and _iot\\test-mails"


# In[20]:


def load(file_name):
    stream=gzip.open(file_name,"rb")
    model=cPickle.load(stream)
    stream.close()
    return model


# In[21]:


def save(file_name,model):
    stream=gzip.open(file_name,"wb")
    cPickle.dump(model,stream)
    stream.close()

making dictionary
# In[22]:


def make_dictionary(root_dir):
    all_words=[]
    emails=[os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words=line.split()
                all_words+=words
    dictionary=Counter(all_words)  #count all words
    list_to_remove=list(dictionary)   #cleaning data
    for item in list_to_remove:
        if item.isalpha()==False:    
            del dictionary[item]
        elif len(item)==1:
            del dictionary[item]
    dictionary=dictionary.most_common(3000)
    return dictionary
        
        


# In[ ]:





# In[23]:


def extract_features(mail_dir):
	files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
	features_matrix = np.zeros((len(files), 3000))
	train_labels = np.zeros(len(files))
	count = 0;
	docID = 0;

	for fil in files:
		with open(fil) as fi:
			for i, line in enumerate(fi):
				if i == 2:
					words = line.split()
					for word in words:
						wordID = 0

						for i, d in enumerate(dictionary):
							if d[0] == word:
								wordID = i
								features_matrix[docID,wordID] = words.count(word)

			train_labels[docID] = 0;
			filepathTokens = fil.split('/')
			lastToken = filepathTokens[len(filepathTokens) - 1]
			# if lastToken.startwith("spmsg")              For Python 2.x
			if lastToken.__contains__("spmsg"):
				train_labels[docID] = 1;
				count = count + 1
			docID = docID + 1

	return features_matrix, train_labels


# In[24]:


dictionary=make_dictionary(Train_dir)
print("Reading and processing from file")


# In[25]:


feature_matrix,labels =extract_features(Train_dir)


# In[26]:


test_feature_matrix,test_labels=extract_features(Test_dir)


# In[27]:


model=svm.SVC(kernel='rbf',C=100,gamma=0.001)
print("trainning model")
model.fit(feature_matrix,labels)
predicted_labels=model.predict(test_feature_matrix)
print("accuracy score")
print(accuracy_score(test_labels,predicted_labels))


# In[28]:


labels.shape


# In[29]:


import os
import numpy as np 
from collections import Counter

from sklearn import svm 
from sklearn.metrics import accuracy_score 

#for 2.x "import cPickle"
import pickle
import gzip

TRAIN_DIR="D:\\machine_learning_and _iot\\train-mails"
TEST_DIR="D:\\machine_learning_and _iot\\test-mails"


# In[30]:


def load(file_name):
	#load the model
	stream = gzip.open(file_name, "rb")
	model = cPickle.load(stream)
	stream.close()
	return model


# In[31]:


def save(file_name, model):
	#save the model
	stream = gzip.open(file_name, "wb")
	cPickle.dump(model, stream)
	stream.close()



# In[32]:



def make_Dictionary(root_dir):
	all_words = []
	emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
	for mail in emails:
		with open(mail) as m:
			for line in m:
				words = line.split()
				all_words += words
	dictionary = Counter(all_words)
	list_to_remove = list(dictionary)
	# list_to_remove = dictionary.keys()     For Python 2.x

	for item in list_to_remove:
		if item.isalpha() == False:
			del dictionary[item]
		elif len(item) == 1:
			del dictionary[item]
	dictionary = dictionary.most_common(3000)

	return dictionary


# In[33]:


def extract_features(mail_dir):
	files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
	features_matrix = np.zeros((len(files), 3000))
	train_labels = np.zeros(len(files))
	count = 0;
	docID = 0;

	for fil in files:
		with open(fil) as fi:
			for i, line in enumerate(fi):
				if i == 2:
					words = line.split()
					for word in words:
						wordID = 0

						for i, d in enumerate(dictionary):
							if d[0] == word:
								wordID = i
								features_matrix[docID,wordID] = words.count(word)

			train_labels[docID] = 0;
			filepathTokens = fil.split('/')
			lastToken = filepathTokens[len(filepathTokens) - 1]
			# if lastToken.startwith("spmsg")              For Python 2.x
			if lastToken.__contains__("spmsg"):
				train_labels[docID] = 1;
				count = count + 1
			docID = docID + 1

	return features_matrix, train_labels


# In[34]:


dictionary = make_Dictionary(TRAIN_DIR)
print("Reading and Processing emails from file.")

features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

model = svm.SVC(kernel="rbf", C=100, gamma=0.001)

print("Traing Model.")

model.fit(features_matrix, labels)

predicted_labels = model.predict(test_feature_matrix)
print(predicted_labels)
print("Finished classifying, accuracy Score: ")
print(accuracy_score(test_labels, predicted_labels))


# In[ ]:




