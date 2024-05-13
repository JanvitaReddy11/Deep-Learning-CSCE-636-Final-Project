import numpy as np
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    
    image = np.transpose(depth_major, [1, 2, 0])

    
    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 1, 0])
   
    return image/255

    ### END CODE HERE



def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE

    if training:
      
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='edge')
        
        

        # Randomly crop a [32, 32] section of the image.
        random_pointh = np.random.randint(9)
        random_pointw = np.random.randint(9)
        range_height = random_pointh + 32
        range_width = random_pointw + 32
        image = image[random_pointh:range_height, random_pointw:range_width, :]

        
        if np.random.rand() > 0.5:
            image = np.fliplr(image)

    # Standardize the image
    

    return image


def visualize(image, save_name='Images/train.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    image = image.reshape((3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    
    image = image.astype(int)
    #print(image)
    
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    #return image

# Other functions
### YOUR CODE HERE

### END CODE HERE