"""
Group 3 (Cody Hartman, Jordan Gahan, Matt Dennie)
Machine Learning 
5/13/17
k-fold validation using custom non-smo SVM.
Feature extraction done using transfer learning with inception v3 CNN.
"""

from math import exp
import numpy as np
import os
import random
import pickle
from svm_project import SVM, kernel_linear
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np


## read training directory
TRAIN_DIR = '../train/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
random.shuffle(train_images)

		
## number of folds
k = 5


## lists to hold folds 
biglabels = []
bigsamples = []
##

## size of the batches
batch_size = 0

	
## load inception model		
def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.
 
    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
 
## extract features from images
def extract_features(image_paths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images
 
    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    ## size of image features
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))
	## open model
    with tf.Session() as sess:
		## select last pooling layer
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		## feed in images to model
        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))
 
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', i)
			## read file
            image_data = gfile.FastGFile(image_path, 'rb').read()
            ## extract features from pool layer
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            ## flatten
            features[i, :] = np.squeeze(feature)
 
    return features 
create_graph("inception_dec_2015/tensorflow_inception_graph.pb")
print("Graph Loaded")
features = extract_features(train_images,True)

## TODO - Incorporate the k-fold testing in the SVM class 
## K-Fold split images into k lists append to big lists
## could also split the list and use less memory.
for i in range(0,k):
	## size of fold 
	batch_size = len(train_images)/k
	## temp lists
	images = []
	labels = []
	## index
	index = (i+1)*batch_size
	## extract labels for images
	for j in range(i*batch_size,index):
		if 'dog' in train_images[j]:
			labels.append(1)
		
		else:
			labels.append(-1)
	## append images by splitting list
	images = (features[batch_size*i:batch_size*(i+1)])
	## add sublist to list
	biglabels.append(labels)
	bigsamples.append(images)

## best accuracy
best = 0.
## avg accuracy
average = 0.
## best model
fin_model = None

for i in range(0, k):	
	## SVM model, C was chosen through testing
	model = SVM(10,kernel_linear)
	#temp sample used for feature extraction
	## concatenate the training lists
	temp_sample = []
	
	temp_label = []
	## will be one list
	validation_label = []
	validation_sample = []
	
	for j in range(0, k):
		if j != i:
			## concat arrays
			temp_sample.extend(bigsamples[j])
			temp_label +=  biglabels[j]
		else:
			
			## should be one list
			validation_sample.extend(bigsamples[j])
			validation_label +=  biglabels[j]
			
	
	

	## convert to numpy array
	validation_sample = np.asarray(validation_sample)
	temp_sample = np.asarray(temp_sample)
	temp_label = np.asarray(temp_label)
	temp_label = temp_label.astype(float)
	
	validation_label = np.asarray(validation_label)
	validation_label = validation_label.astype(float)

	## Start training the model 
	model.fit(temp_sample,temp_label)	
	## result of validation
	result = 0.
	for l in range(0, len(validation_sample)):
		## predict on validation samples
		predict = model.predict(validation_sample[l])
		print (float(np.sign(predict).item()),float(validation_label[l]))
		## if prediction is correct, then add to result
		if float(np.sign(predict).item()) == float(validation_label[l]):
			
			result += 1.0
	## calculate the result
	result = result/len(validation_sample)
	
	## add to average
	average += result
	## if best
	if result > best:
		## set best accuracy
		best = result
		## set best model
		fin_model = model
	## print accuracy
	print(result)
## print best accuracy
print('The best model had an accuracy of ', best)	
## print avg accuracy	
print('Average Accuracy of ', average/k)	

## save model
f1 = open('model.fn','wb')
pickle.dump(fin_model,f1)
f1.close()

