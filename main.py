### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split,load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_src", default="Data/cifar-10-batches-py")
parser.add_argument("--data_dir", default="Data2024/private_test_images_2024.npy")
parser.add_argument("--save_dir", default="models1")
parser.add_argument("--result_dir", default="results/predictions.npy")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(training_configs)

	

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		#model.train(x_train, y_train, training_configs, x_valid = None, y_valid = None)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		#model.train(x_train, y_train, training_configs, x_valid, y_valid)

		model.evaluate(x_valid, y_valid,[60,70,80,90,100,110,130,150,160,170,180,190,200])

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test,[90,100,160,170,180,190,200])

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test)
		np.save(args.result_dir, predictions)
		

### END CODE HERE

