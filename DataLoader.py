import os
import pickle
import numpy as np
from PIL import Image

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    files = os.listdir(data_dir)
    
    
    for i in range(len(files)):
        if 'data_batch' in files[i]:
            with open(os.path.join(data_dir, files[i]), 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            x_train.append(data[b'data'])
            y_train.extend(data[b'labels'])
        elif files[i] == 'test_batch':
            with open(os.path.join(data_dir, files[i]), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            x_test.append(data[b'data'])
            y_test.extend(data[b'labels'])

    # Convert lists to arrays
    x_train = np.concatenate(x_train,axis = 0)
    x_test = np.concatenate(x_test,axis = 0)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

 
   
    
   


    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

    ### END CODE HERE

    #return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    
    images = []
    
    files = os.listdir(data_dir)

    for file in files:
        # Load the image
        image_path = os.path.join(data_dir, file)
        image = Image.open(image_path)
        # Convert the image to numpy array and flatten it
        image = np.array(image).flatten()
        # Append the flattened image to the list
        images.append(image)

    # Convert the list of images to numpy array
    x_test = np.array(images, dtype=np.float32)
    #return x_test
    
    
    

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = int(x_train.shape[0]*train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    #return x_train_new, y_train_new, x_valid, y_valid

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

