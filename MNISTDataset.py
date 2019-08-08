#This code was partly inspired by Syafiq Kamarul Azman's article on datasets here: 
#https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

from torch.utils.data import Dataset
import gzip
import numpy as np
import torch
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, image_data_root, label_data_root):
        #Variables for the image set
        self.image_data_root = image_data_root
        self.image_magic_number = 0
        self.num_images = 0
        self.image_rows = 0
        self.image_columns = 0
        self.images = np.empty(0)
        
        #Variables for labels
        self.label_data_root = label_data_root
        self.label_magic_number = 0
        self.num_labels = 0
        self.labels = np.empty(0)
        
        #Functions that initialize the data
        self.image_init_dataset()
        self.label_init_dataset()
        
        
    #Returns the number of images in the set
    def __len__(self):
        return self.num_images
    
    #Returns an image based on a given index
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    #This method gets the images from the MNIST dataset
    def image_init_dataset(self):
        #Unzips the image file.  'r' is the mode argument and is just telling it to read the data in binary mode instead of 
        #as text
        image_file = gzip.open(self.image_data_root, 'r')
        #Datatype that switches the byteorder for the dataset.  The original data is in high-endian byte order, however if
        #you are using a computer with an Intel CPU, your computer may be working by default in low-endian byte order.  This
        #step guarantees you will not have issues with reading the data.  If you were to not do this step, you would get the
        #wrong numbers
        #Sources: https://stackoverflow.com/a/39596773  https://stackoverflow.com/a/53226079
        #The next set of bytes will be read as 32 bit ints so the dtype here is int32
        reorder_type = np.dtype(np.int32).newbyteorder('>')
        #The first 16 bytes of the file are used to give a magic number, the number of images, the rows for each
        #image, as well as the columns for each.  Each number for these is a 32 bit int and the byte order is
        #changed, thus the reorder type is used.  If the byteorder was not changed, the bytes would be in reverse
        #order, giving very different numbers
        self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        
        #Now that the first 16 bytes are done, we have to get the rest of the bytes out and organized into a numpy array
        #Here we have the buffer read the total bytes for the images by reading 60000*28*28 bytes.  Each image is 28x28
        #pixels, with each pixel represented by an 8 bit integer (1 byte).  The 28x28 is then multiplied by
        #the number of images in the set, which is 60000, to get the total number of bytes we need to read
        buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)
        #Next we read the bytes from the buffer as unsigned 8 bit integers (np.uint8), and then put them into a
        #numpy array as 32 bit floats.  This is now a 1D array (a flattened vector) of all the data
        self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        #Here we make the 1D array into a 60000x784 array to be useable with neural networks
        self.images = np.reshape(self.images, (self.num_images, 784))
        #This normalizes the data to be between 0 and 1.  The 255 is the range of the pixel values
        self.images = self.images/255
        #Turns the data to tensors as that is the format the neural networks use
        self.images = torch.tensor(self.images)
        
        
    #This method gets the labels from the MNIST dataset
    def label_init_dataset(self):
        label_file = gzip.open(self.label_data_root, 'r')
        
        reorder_type = np.dtype(np.int32).newbyteorder('>')
        #As only the magic number and number of labels are given here, we will only do these two
        self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        self.num_labels = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        
        buffer = label_file.read(self.num_labels)
        #We leave this as a 1D array as labels do not have any further dimensions
        self.labels = np.frombuffer(buffer, dtype = np.uint8)
        #The datatype is 'long' here as the loss function for the neural network, Cross Entropy Loss,
        #is made to work with longs
        self.labels = torch.tensor(self.labels, dtype = torch.long)
        
#Testing function that draws images from the datasets based on a given index
def draw_image(images_root, labels_root, image_idx):
    mnist = MNISTDataset(images_root, labels_root)
    #Here, we reshape the 1D array into a 60000x28x28x1 dimensional array.  This will allow us to be
    #able to pull individual images and read them
    mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
    image, label = mnist.__getitem__(image_idx)
    print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
    print('Label: {}'.format(label.item()))
    plt.imshow(image)
    plt.show()
    
if __name__ == "__main__":
    #This will draw an image from each dataset depending on the index you input.  Will only be called if
    #you call this class directly
    draw_image('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz', 500)
    draw_image('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz', 500)


        
        