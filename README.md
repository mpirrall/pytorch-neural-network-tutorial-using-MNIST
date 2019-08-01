# Pytorch Neural Network and Dataset Tutorial Using MNIST

This tutorial will cover creating a custom Pytorch Dataset with the MNIST dataset and using it to train a basic feedforward neural network in Pytorch.  There will be four main parts: extracting the MNIST data into a useable form, creating the neural network itself, training the network, and, lastly, testing it.

## Introduction

Pytorch is a dataset of handwritten images, often considered the 'Hello world!' of machine learning.  It is composed of 70,000 images, which are further split into 60,000 images designated for training neural networks and 10,000 designated for testing them.  While many tutorials for MNIST use a set that has already been preprocessed, this one will cover how to load the set from the raw files.  This method was chosen to give more experience with loading actual datasets, as most real world data is not neatly processed for you.  With that said, let's begin!

### What You Need
Installing the following programs/packages can be difficult for a beginner.  I'll try to point you in the right direction, but if you are having trouble then you should look up more in-depth tutorials on that specific topic online before moving forward.

Anaconda and a console emulator:

* I would recommend having [Anaconda](https://www.anaconda.com/distribution/) installed as it is great for downloading packages and just doing anything data science related in general.  I will be moving forward in this section assuming you have Anaconda.
* If you're on Windows and need a console emulator to work with Anaconda, I'd highly recommend [Cmder](https://cmder.net/).

Once you have Anaconda installed, you'll need the following packages:
* [Pytorch](https://pytorch.org/) - Scroll down and customize based on your system.  Choose *Conda* as you are using Anaconda, and if you don't know what CUDA is, choose *None*.  Run what you're given in your console (Cmder if you chose to use it) to download Pytorch.
* [Matplotlib](https://anaconda.org/conda-forge/matplotlib)- Follow the link and run one of the commands in console.
* Numpy- You shouldn't actually need to install this as it should come with Anaconda.

Once you have finished installing them, you can confirm you have them by running **conda list** in console and finding the packages in the list.

MNIST:
* We'll be using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for training and testing our neural network, so download the four files named **train-images-idx3-ubyte.gz**, **train-labels-idx1-ubyte.gz**, **t10k-images-idx3-ubyte.gz**, and **t10k-labels-idx1-ubyte.gz** found near the top of the linked site.  As it says, these are the images and labels for training and testing.
* Once the files are downloaded, extract them into an easy to find folder for later.

With that finished, we should be ready to code!

## Creating the Dataset
We'll begin by creating the custom Pytorch Dataset that will hold the MNIST data.  This class will be general enough that it can be used for both the training data and testing data.  The Dataset class from Pytorch is used so that we can put the dataset into a Pytorch DataLoader, which will then be able to feed the data into our neural network.  I know it sounds complicated, but it will make sense once you see it implemented.

### Dataset - Imports
First, we have to import all the packages we'll need for this class:

    from torch.utils.data import Dataset
    import gzip
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

Package descriptions:
* The Dataset class is what we will be extending to create our own custom Dataset.  As mentioned before, by extending this class to make our own, we will be able to feed this Dataset into the Pytorch DataLoader class, which will make training much easier.
* gzip will be used to unzip the MNIST files you downloaded earlier.  As you may have noticed, they are all .gz files, which is the extension for files compressed by gzip.
* numpy is great for making multidimensional arrays and just holding data in general.  It's better than using the Python Lists as it uses less memory and is faster with calculations, as well as making it easy to change the dimensions of the data.
* torch will be used for this class to turn the data from numpy arrays into tensors, which is the format neural networks use for data.
* Matplotlib (specifically pyplot in this case) is a package that allows you to easily make visualizations of data.  We will be using it to double check that our digits are displaying correctly.

Now that the packages are imported, we can move into actually creating our custom Dataset!

### Dataset - \_\_init\_\_, \_\_len\_\_, and \_\_getitem\_\_
To make our class a subclass of the Pytorch Dataset class (extending it) we must put it within the class parentheses like so:

    class MNISTDataset(Dataset):
    
With that one, we must now create the \_\_init\_\_, \_\_len\_\_, and \_\_getitem\_\_ functions for the Dataset class.  Overriding \_\_len\_\_ and \_\_getitem\_\_ is necessary in all child classes of the Pytorch Dataset class as that is what will be called by the Pytorch Dataloader to get values from the dataset.

### Dataset - \_\_init\_\_
The init is the constructor for our dataset class.  As the training and testing sets each have a set of images and a set corresponding labels, we will take the root (file location) of the image set and label set as arguments:

    def __init__(self, image_data_root, label_data_root):

Remember that as we are going to make two instances of this class, one for the training data and one for the testing data, we only need one root for the images and one root for the labels each time we create an instance of this class.

Within the init function we are going to put all the variables we will need throughout the class, as well as function calls to format the data.  I prefer to do all the computation in separate functions that are called from the init as it not only makes it easier to read, but also makes it easier to find bugs or other issues.

Before we make the variables, I want to explain the format of the MNIST files.  While it is explained on the [MNIST site](http://yann.lecun.com/exdb/mnist/) near the bottom of the page under the section headed "**FILE FORMATS FOR THE MNIST DATABASE**", their explanation can be quite confusing.  I will do my best to explain it as easily and concisely as possible. 

#### Dataset - MNIST File Format
*Feel free to skim this section if you aren't particularly interested in the file format as this is mainly to help with better understanding.  I'll reiterate the critical parts later.*

The idx#-ubyte formats are simply a long, one dimensional vector of bytes.  While I could not find any confirmation for this, I think the 1-ubyte or 3-ubyte in the file format stand for the amount (1 or 3) of 32-bit integers at the beginning of the file when starting from 0 (so 1 or 3 from 0 means 2 or 4 total integers), and the data type (ubyte, or unsigned byte).  Also, the format begins with integers, with the rest of the bytes being single unsigned bytes. Knowing that, I'll quickly give the characteristics of the sets:

Image sets (...-images-idx3-ubyte):
* Begins with 4 32-bit integers, as explained above.  Keep in mind each integer is 4 bytes for a total of 16 bytes.
* First integer is magic number, look at bottom of the [MNIST page](http://yann.lecun.com/exdb/mnist/) under **THE IDX FILE FORMAT** for explanation.
* Second integer is the number of images in the set (either 10,000 or 60,000).
* Third integer is the number of rows in image (28).
* Fourth integer is the number of columns in the image (28).
* The rest of the file (bytes 16 and beyond) is made up of single unsigned bytes (value between 0-255) each representing a single pixel of the image.
* As the image is 28x28 pixels (and 28x28 = 784), every 784 bytes (pixels) is a single image.  This will be important later when we extract the images.

Label sets (...-labels-idx1-ubyte):
* Begins with 2 32-bit integers.  Each integer is 4 bytes for a total of 8 bytes.
* First integer is the magic number.
* Second integer is the number of labels (either 10,000 or 60,000).
* Rest of the file (bytes 8 and beyond) is made up of single unsigned bytes valued between 0 and 9, giving the correct number for it's index in the corresponding image file.

Whew.. With that done, let's get back to coding.

### Dataset - \_\_init\_\_ (continued)

Now that we know the contents of the files, we can make variables for all of them.  I'll include what we've previously done with this class as well for better understanding:

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

                #Functions that intialize the data
                self.image_init_dataset()
                self.label_init_dataset()

For the image set:
* Saved root to open file later.
* Intialized the magic number, number of images in the set, number of rows in each image, and number in columns in each image, which are the first 4 integers in the file. The 0's will all be replaced later with the actual numbers when we get them from the file.
* Initialized an empty numpy array which we can later replace with the full array of images.

For the label set:
* Saved root for file
* Initialized the magic number and number of labels.
* Intialized an empty numpy array to replace later with the labels.

Functions:
* The image_init_dataset() function will be called to process the image file
* The label_init_dataset() function will process the label file

With that the \_\_init\_\_ is finished.  I promise we'll get to fun stuff soon, just trust the process.

### Dataset - \_\_len\_\_ and \_\_getitem\_\_
The length and getitem functions are required as we are extending the Pytorch Dataset class.  Luckily they are very easy to implement.  \_\_len\_\_ just requires us to return the length of the dataset, which we can pull directly from the image set file as it is one of those first 4 integers I mentioned earlier.  \_\_getitem\_\_ requires us to return a single image and label for each call based on a given index, which is simply getting values from an array. 

We can implement these like so:

    #Returns the number of images in the set
    def __len__(self):
        return self.num_images
    
    #Returns an image based on a given index
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

Now just one more thing before we get to move into the fun stuff...

### Dataset - Making a Main Function for Testing
As we continue to make this dataset, you may want to test what you've done so far to see if it works.  With this is mind, we're going to make a small function that we will update as we make progress through the tutorial so you can test as you go.  First, you're going to create a function outside of the class we've been working on (no indent) called draw_image:

    def draw_image(images_root, labels_root, image_idx):
        mnist = MNISTDataset(images_root, labels_root)

We will actually have it draw an image later, but for now we just want it to try to build our dataset.  To build the dataset, we need a the roots (file locations) for the dataset and label, as well as an index for the image we want to draw.  We then create the dataset by creating an object with our MNISTDataset class.  draw_image is just a function, however.  It won't run without a main, so we need to create one to call it:

    if __name__ == "__main__":
        draw_image('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz', 300)
        draw_image('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz', 300)
        
By using the " if \_\_name\_\_ == "\_\_main\_\_": " format, the draw_image function will only be called when running the MNISTDataset.py file.  It won't be called when we import this file later to use with the neural network.  If you want to know more, check out (this link)[https://stackoverflow.com/a/419185/11788566] for a great explanation as to how it works.  Instead of the file locations I have there, replace them with wherever you stored your files.  Make sure the files themselves match up though!  The index of 300 is just arbitrary, and you could make it anywhere between 0 and 59999 for the training set or 0 and and 9999 for the test set, as that is how many images they each have.

To make sure there's no confusion, here is what the format of your code should look like so far: 

    from torch.utils.data import Dataset
    # other imports...

    class MNISTDataset(Dataset):
        def __init__(self, image_data_root, label_data_root):
            # variables...

        def __len__(self):
            return self.num_images
            
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    def draw_image(images_root, labels_root, image_idx):
        mnist = MNISTDataset(images_root, labels_root)

    if __name__ == "__main__":
        draw_image('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz', 300)
        draw_image('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz', 300)
        
If you tried to run this, you'll notice it didn't work.  Don't worry, we'll fix that soon.

### Dataset - Getting the Image Data
The next function we will make is the image_init_dataset() function that is called in the \_\_init\_\_ function to process the file with the image set.  

As this may be difficult to follow along with, I'm going to give all the code for this function first, and then explain it piece by piece.  I wouldn't recommend copying all this intial code right away, rather I'm putting it here for you to look back at if you want to see how the the current section I'm working on fits into the bigger picture. Here's the full function:

    #This method gets the images from the MNIST dataset
    def image_init_dataset(self):
        #Unzips the image file
        image_file = gzip.open(self.image_data_root, 'r')
          
        #Datatype that switches the byteorder for the dataset
        reorder_type = np.dtype('int32').newbyteorder('>')
            
        #Getting the first 16 bytes from the file(first 4 32-bit integers)
        self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
            
        #Getting all the bytes for the images into the buffer
        buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)
            
        #Next we read the bytes from the buffer as unsigned 8 bit integers (np.uint8), and then put them into a
        #numpy array as 32 bit floats.  This is now a 1D array (a flattened vector) of all the data
        self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
            
        #Here we make the 1D array into a 60000x784 array (images are flattened) to be useable with neural networks
        self.images = np.reshape(self.images, (self.num_images, 784))
            
        #This normalizes the data to be between 0 and 1.  The 255 is the range of the pixel values (0-255)
        self.images = self.images/255
          
        #Turns the data to tensors as that is the format that neural networks use
        self.images = torch.from_numpy(self.images)

Section by setion:

    def image_init_dataset(self):
        image_file = gzip.open(self.image_data_root, 'r')
        
After we create the function itself, we have to unzip the file so we can pull the data from it.  As I said back in the imports section, we are working with .gz files which mean we must use gzip to open it.

    reorder_type = np.dtype('int32').newbyteorder('>')

The above line may be confusing depending on how much you know of computer science.  This data is stored in a big-endian byte format, which is type of byte storage format.  Some CPU's read bytes as big-endian and some as little-endian.  Typically, the big-endian format is used by non-Intel processors, while Intel processors use little-endian.  This byte order should make it work no matter what, however if you are getting getting numbers that aren't 60000, 10000, or 28 for the initial integers, try switching the byte order the other way by changing the '>' to a '<'.

    self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        
Each of the above lines reads four bytes from the file, where each set of bytes is a 32-bit integer (4 x 8 bits in a byte = 32).  To make sure the number we get is correct, we read them with the reorder_type we just created. Lastly, as we are reading it using a numpy function, it will store the data we read into a numpy array. We want the integer itself rather than an array with 1 number, so we use [0] to store only the integer.

    buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)
        
Now we are going to read all the image data from the file into a temporary variable that we'll call 'buffer'.  As you know, each byte is a pixel, so we want to read all 60,000 (if it's the training dataset) or 10,000 (if it's the testing dataset) 28x28 pixel images.  To do this, we will use the data we just got about the number of images, row size, and column size to determine the number of bytes we read.

    self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)

This line now reads all the data from the buffer variable as unsigned ints (np.unit8) and stores them into a numpy array as 32-bit floats (np.float32).  This is now our set of image data.  Currently the data is stored as as really long one-dimensional set of numbers.  To see this, you can add print(self.images.shape) like so:

    self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
    print(self.images.shape)

and run the program. You'll get an error, but if you check above it said error, you should see an output that looks like **(47040000,)**.  These are the 60000x28x28 pixels that you've extracted (from now on, know that if I say 60,000 I'm talking about the training set and 10,000 is the testing set).  We'll change the dimensions of it so they are separated into individual images next. As we continue, you can add that print statement we just used wherever you want to continue to check the shape of self.images.

    self.images = np.reshape(self.images, (self.num_images, 784))

Here we reshape that long, one-dimensional set of numbers into a set of individual images.  np.reshape takes as arguments the numpy array you're modifying, and then a tuple of the dimensions you want the new one to be.  It then returns the newly reshaped array.  Thus we are reshaping self.images (the first argument) and changing its dimensions to 60000x784 (the second argument).  784 is the number of pixels in the rows and columns (28x28=784) for each image.  As we will be making a feedforward neural network, we want the images in this set to be in a flattened state of 784 bytes.  This is because we will feed the set of bytes into the neural network rather than have it look at the whole "image". 

    self.images = self.images/255

This line will divide all the pixels (which are in greyscale between 0-255) by 255 to normalize them between 0 and 1.  We will be using a sigmoid activation function for our neural network later, which looks like this: 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg">
  <br>Source: https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg <br>
</p>

As you can see, the range of the sigmoid function where the derivative (slope) changes the most is between 0 and 1.  Thus, learning can quickly occur here.  Having values much larger or smaller than that range can cause neural network saturation
