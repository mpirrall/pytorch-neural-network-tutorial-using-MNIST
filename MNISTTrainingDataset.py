#The MNIST dataset is composed of a long series of bits.  For the image sets, the first 16 bits give non-image data
#For the label sets, the first 8 bits are non-label data

from torch.utils.data import Dataset
import gzip
import numpy as np
import torch
import matplotlib.pyplot as plt


class MNISTTrainingDataset(Dataset):
    def __init__(self, image_data_root, label_data_root):
        self.image_data_root = image_data_root
        self.image_magic_number = 0
        self.num_images = 0
        self.image_rows = 0
        self.image_columns = 0
        self.training_images = np.empty(0)
        
        #Variables for labels
        self.label_data_root = label_data_root
        self.label_magic_number = 0
        self.num_labels = 0
        self.training_labels = np.empty(0)
        
        #Functions that intialize the data
        self.image_init_dataset()
        self.label_init_dataset()
        
        #Creates the value list for __getitem__
        #self.value_list = list(range(0, self.num_images))
        
    #Returns the number of images in the set
    def __len__(self):
        return self.num_images
    
    #Returns an image based on an index
    def __getitem__(self, idx):
        return self.training_images[idx], self.training_labels[idx]
    
    #This method gets the images from the MNIST dataset
    def image_init_dataset(self):
        #Unzips the training file
        training_file = gzip.open(self.image_data_root, 'r')
        #Datatype that switches the byteorder for the dataset.  Source: https://stackoverflow.com/a/53226079
        #The next set of bytes will be read as 32 bit ints so the dtype here is int32
        #This is done as I have an intel CPU, you may not have to do this
        reorderType = np.dtype('int32').newbyteorder('>')
        #The first 16 bytes of the file are used to give a magic number, the number of images, the rows for each
        #image, as well as the columns for each.  Each number for these is a 32 bit int and the byte order is
        #changed, thus the reorder type is used.  If the byteorder was not changed, the bytes would be in reverse
        #order, giving very different numbers
        self.image_magic_number = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        self.num_images = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        self.image_rows = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        self.image_columns = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        
        #Now that the first 16 bytes are done, we have to get the rest of the bytes out and organized into a numpy array
        #Here we have the buffer read the total bytes for the images by reading 60000*28*28 bytes.  Each image is 28x28
        #pixels, with each pixel represented by an 8 bit integer (1 byte).  The 28x28 is then multiplied by
        #the number of images in the set, which is 60000, to get the total number of bytes we need to read
        buffer = training_file.read(self.num_images*self.image_rows * self.image_columns)
        #Next we read the bytes from the buffer as unsigned 8 bit integers (np.uint8), and then put them into a
        #numpy array as 32 bit floats.  This is now a 1D array (a flattened vector) of all the data
        self.training_images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        #Here we make the 1D array into a 60000x784 array to be useable with neural networks
        self.training_images = np.reshape(self.training_images, (self.num_images, 784))
        #This normalizes the training data to be between 0 and 1.  The 255 is the range of the pixel values
        self.training_images = self.training_images/255
        #Turns the data to tensors as that is the format the pytorch neural networks use
        self.training_images = torch.from_numpy(self.training_images)
        
        #UNCOMMENT THIS CODE WHEN SHOWING IMAGES ON CONSOLE. KEEP COMMENTED WHEN USING NEURAL NETWORK
        #Here, we reshape the the 1D array into a 60000x28x28x1 dimensional array.  This will allow use to be
        #able to pull individual images and read them
        #self.training_images = np.reshape(self.training_images, (self.num_images, 28, 28, 1))
        
    #This method gets the labels from the MNIST dataset
    def label_init_dataset(self):
        training_file = gzip.open(self.label_data_root, 'r')
        
        reorderType = np.dtype('int32').newbyteorder('>')
        #As only the magic number and number of labels are given here, we will only do these two
        self.label_magic_number = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        self.num_labels = np.frombuffer(training_file.read(4), dtype=reorderType).astype(np.int64)[0]
        
        buffer = training_file.read(self.num_labels)
        #We leave this as a 1D array as labels do not have any further dimensions
        self.training_labels = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        
        #The datatype is 'long' here as the loss function for the neural network, Cross Entropy Loss,
        #is made to work with longs
        self.training_labels = torch.tensor(self.training_labels, dtype = torch.long)
        


#UNCOMMENT THIS CODE TO SHOW IMAGES ON CONSOLE. KEEP COMMENTED WHEN USING NEURAL NETWORK
#mnist = MNISTTrainingDataset('C:/Easy/train-images-idx3-ubyte.gz', 'C:/Easy/train-labels-idx1-ubyte.gz')
#The number you place here will get the image from the dataset
#image, label = mnist.__getitem__(50000)
#image = image.squeeze()
#print(image.shape)
#print(label)
#plt.imshow(image)
#plt.show()

        
        