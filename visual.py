from svm_project import SVM, kernel_linear

from PIL import Image, ImageFilter, ImageFont, ImageDraw

from math import exp
import numpy as np
import os
import random
import pickle
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
from Tkinter import Tk, Label, Button
import tkFileDialog



f1 = open('model.fn','rb')
model = pickle.load(f1)
f1.close()



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
 
 
def extract_features(image_paths, verbose=False):
	"""
	extract_features computed the inception bottleneck feature for a list of images
 
	image_paths: array of image path
	return: 2-d array in the shape of (len(image_paths), 2048)
	"""
	feature_dimension = 2048
	features = np.empty((len(image_paths), feature_dimension))
 
	with tf.Session() as sess:
		flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
 
		for i, image_path in enumerate(image_paths):
			if verbose:
				print('Processing %s...' % (image_path))
 
			if not gfile.Exists(image_path):
				tf.logging.fatal('File does not exist %s', image)
 
			image_data = gfile.FastGFile(image_path, 'rb').read()
			feature = sess.run(flattened_tensor, {
				'DecodeJpeg/contents:0': image_data
			})
			features[i, :] = np.squeeze(feature)
 
	return features 
create_graph("inception_dec_2015/tensorflow_inception_graph.pb")
print("Graph Loadeds")




class MyFirstGUI:
	def __init__(self, master):
		self.master = master
		master.title("A simple GUI")

		self.label = Label(master, text="Select Some Images!!")
		self.label.pack()

		self.greet_button = Button(master, text="Open Files", command=self.files)
		self.greet_button.pack()

		self.close_button = Button(master, text="Close", command=master.quit)
		self.close_button.pack()

	def files(self):
		animal = {-1.0:'cat',1.0:'dog'}
		filez = tkFileDialog.askopenfilenames(parent=self.master,title='Choose a file')
		images = root.tk.splitlist(filez)
		features = extract_features(images)
   
		
		for i in range(len(images)):
			result = np.sign(model.predict(features[i]))
			result = animal[result]
			img = Image.open(images[i])
			draw = ImageDraw.Draw(img)
			font = ImageFont.truetype("arimo/Arimo-Bold.ttf", 32)
			draw.text((0, 0), result, font=font, fill=(255,0,0,255) )
			img.show()

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()

