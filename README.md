# Pytorch Neural Network and Dataset Tutorial Using MNIST

This tutorial will cover creating a custom Pytorch Dataset with the MNIST dataset and using it to train a basic feedforward neural network in Pytorch.  There will be four main parts: extracting the MNIST data into a useable form, creating the neural network itself, training the network, and, lastly, testing it.

## Introduction

Pytorch is a dataset of handwritten images, often considered the 'Hello world!' of machine learning.  It is composed of 70,000 images, which are further split into 60,000 images designated for training neural networks and 10,000 designated for testing them.  While many tutorials for MNIST use a set that has already been preprocessed, this one will cover how to load the set from the raw files.  This method was chosen to give more experience with loading actual datasets, as most real world data is not neatly processed for you.  With that said, let's begin!

### What you need
Installing the following programs/packages can be difficult for a beginner.  I'll try to point you in the right direction, but if you are having trouble then you should look up more in-depth tutorials on that specific topic online before moving forward.

Anaconda and a console emulator:

* I would recommend having [Anaconda](https://www.anaconda.com/distribution/) installed as it is great for downloading packages and just doing anything data science related in general.  I will be moving forward in this section assuming you have Anaconda.
* If you're on Windows and need a console emulator to work with Anaconda, I'd highly recommend [Cmder](https://cmder.net/).

Once you have Anaconda install:
* [Pytorch](https://pytorch.org/) - Scroll down and customize based on your system.  Choose *Conda* as you are using Anaconda, and if you don't know what CUDA is, choose *None*.  Run what you're given in your console (Cmder if you chose to use it) to download Pytorch.
* [Matplotlib](https://anaconda.org/conda-forge/matplotlib)- Follow the link and run one of the commands in console.
* Numpy- You probably won't need to actually install this as it should come with Anaconda.

Once you have finished installing them, you can confirm you have them by running **conda list** in console and finding the packages in the list.

MNIST:
* We'll be using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for training and testing our neural network, so download the four files named **train-images-idx3-ubyte.gz**, **train-labels-idx1-ubyte.gz**, **t10k-images-idx3-ubyte.gz**, and **t10k-labels-idx1-ubyte.gz**.  As the website says, these are the images and labels for training and testing.
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
    
* The Dataset class is what we will be extending to create our own custom Dataset.  As mentioned before, by extending this class to make our own, we will be able to feed this Dataset into the Pytorch DataLoader class, which wil make training much easier.
* gzip will be used to unzip the MNIST files you downloaded earlier.  As you may have noticed, they are all .gz files, which is the extension for files compressed by gzip.
* numpy is great for making multidimensional arrays and just holding data in general.  It's better than using the Python Lists as it uses less memory and is faster with calculations, as well as making it easy to change the dimensions of the data.
* torch will be used for this class to turn the data from numpy arrays into tensors, which is the format neural networks use for data.
* Matplotlib (specifically pyplot in this case) is a package that allows you to easily make visualizations of data.  We will be using it to double check that our digits are displaying correctly.

Now that the packages are imported, we can move into actually creating our custom Dataset!

### Dataset - __init__, __len__, and __getitem__
To make our class a subclass of the Pytorch Dataset class (extending it) we must put it within the class parentheses like so:

    class MNISTDataset(Dataset):
    
With that one, we must now create the __init__, __len__, and __getitem__ functions for the Dataset class.  Overriding __len__ and __getitem__ is necessary in all child classes of the Pytorch Dataset class as that is what will be called by the Pytorch Dataloader to get values from the dataset.

#### Dataset - __init__
The init is the constructor for our dataset class.  As the training and testing sets each have a set of images and a set corresponding labels, we will take the root (file location) of the image set and label set as arguments:

    def __init__(self, image_data_root, label_data_root):

Remember that as we are going to make two instances of this class, one for the training data and one for the testing data, we only need one root for the images and one root for the labels each time we create an instance of this class.

Within the init function we are going to put all the variables we will need throughout the class, as well as function calls to format the data.  I prefer to do all the computation in separate functions that are called from the init as it not only makes it easier to read, but also makes it easier to find bugs or other issues.

Before we make the variables, I want to explain the format of the MNIST files.  While it is explained on the [MNIST site](http://yann.lecun.com/exdb/mnist/) near the bottom of the page under the section headed "**FILE FORMATS FOR THE MNIST DATABASE**", their explanation can be quite confusing.  I will do my best to explain it as easily and concisely as possible. 

##### Dataset - MNIST File Format
*Feel free to skip this section if you aren't particularly interested in the file format as this is mainly to help with better understanding.  I'll reference the necessary parts later.*

The idx#-ubyte formats are simply a long, one dimensional vector of bytes.  While I could not find any confirmation for this, I think the 1-ubyte or 3-ubyte in the file format stand for the amount (1 or 3) of 32-bit integers at the beginning of the file when starting from 0 (so 1 or 3 from 0 means 2 or 4 total integers), and the data type (ubyte, or unsigned byte).  With the knowledge that the format begins with integers and then has the rest of the bytes being single unsigned bytes, I'll quickly give the characteristics of the sets:

Image sets (...-images-idx3-ubyte):
* Begins with 4 32-bit integers, as explained above.  Keep in mind each integer is 4 bytes for a total of 16 bytes.
* First integer is magic number, look at bottom of the [MNIST page](http://yann.lecun.com/exdb/mnist/) under **THE IDX FILE FORMAT** for explanation.
* Second integer is the number of images in the set (either 10,000 or 60,000).
* Third integer is the number of rows in image (28).
* Fourth integer is the number of columns in the image (28).
* The rest of the file is made up of single unsigned bytes (value between 0-255) each representing a single pixel of the image.
* As the image is 28x28 pixels (and 28x28 = 784), every 784 bytes (pixels) is a single image.  This will be important later.

Label sets (...-labels-idx1-ubyte):
* Begins with 2 32-bit integers.  Each integer is 4 bytes for a total of 8 bytes.
* First integer is the magic number
* Second integer is the number of labels
* Rest of the file is single unsigned bytes valued between 0 and 9, giving the correct number for it's index in the corresponding image file

Whew.. With that done, let's get back to coding.

#### Dataset - __init__ (continued)




