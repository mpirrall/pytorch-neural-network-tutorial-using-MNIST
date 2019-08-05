# Pytorch Neural Network and Dataset Tutorial Using MNIST

This tutorial will cover creating a custom Pytorch Dataset with the MNIST dataset and using it to train a basic feedforward neural network in Pytorch.  There will be four main parts: extracting the MNIST data into a useable form, creating the neural network itself, training the network, and, lastly, testing it.

This tutorial was created to be in-depth enough that even people with very little experience could understand it.  Many tutorial focus on *what* they did, not *why* they did it.  I try to delve more deeply into the *why* aspect, but due to that the tutorial is a little lengthy.  I will generally put my code at the beginning of each section, so if you feel you understand it, feel free not to read the more in-depth line by line explanations unless you feel you need them.

## Introduction

Pytorch is a dataset of handwritten digits, often considered the 'Hello world!' of machine learning.  It is composed of 70,000 images, which are further split into 60,000 images designated for training neural networks and 10,000 designated for testing them.  Each image is of the size 28x28 pixels and depicts a number from 0-9.  While many tutorials for MNIST use a set that has already been preprocessed, this one will cover how to load the set from the raw files.  This method was chosen to give more experience with loading actual datasets, as most real world data is not neatly processed for you.  With that said, let's begin!

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

We will actually have it draw an image later, but for now we just want it to try to build our dataset.  To build the dataset, we need the roots (file locations) for the dataset and label, as well as an index for the image we want to draw.  We then create the dataset by creating an object with our MNISTDataset class.  draw_image is just a function, however, and won't run without a main, so we need to create one to call it:

    if __name__ == "__main__":
        draw_image('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz', 300)
        draw_image('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz', 300)
        
By using the " if \_\_name\_\_ == "\_\_main\_\_": " format, the draw_image function will only be called when running the MNISTDataset.py file.  It won't be called when we import this file later to use with the neural network.  If you want to know more, check out [this link](https://stackoverflow.com/a/419185/11788566) for a great explanation as to how it works.  Instead of the file locations I have there, replace them with wherever you stored your files.  Make sure the files themselves match up though!  Don't mix up the training and testing sets.  The index of 300 is just arbitrary, and you could make it anywhere between 0 and 59999 for the training set or 0 and and 9999 for the test set, as that is how many images they each have.

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
        
If you try to run this, you'll see it doesn't work.  Don't worry, we'll fix that soon.

### Dataset - Getting the Image Data
The next function we will make is the image_init_dataset() function that is called in the \_\_init\_\_ function to process the file with the image set.  

As this may be difficult to follow along with, I'm going to give all the code for this function first, and then explain it piece by piece.  I wouldn't recommend copying all this intial code right away, rather I'm putting it here for you to look back at if you want to see how the the current section I'm working on fits into the bigger picture. Here's the full function:

    #This method gets the images from the MNIST dataset
    def image_init_dataset(self):
        #Unzips the image file
        image_file = gzip.open(self.image_data_root, 'r')
          
        #Datatype that switches the byteorder for the dataset
        reorder_type = np.dtype(np.int32).newbyteorder('>')
            
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
        
After we create the function itself, we have to unzip the file so we can pull the data from it.  As I said back in the imports section, we are working with .gz files which mean we must use gzip to open it.  The 'r' argument just sets the 'mode' to read the file in binary rather than as text data.

    reorder_type = np.dtype(np.int32).newbyteorder('>')

The above line may be confusing depending on how much you know of computer science.  This data is stored in a big-endian byte format, which is type of byte storage format.  Some CPU's read bytes as big-endian and some as little-endian.  Typically, the big-endian format is used by non-Intel processors, while Intel processors use little-endian.  This byte order should make it work no matter what, however if you are getting getting numbers that aren't 60000, 10000, or 28 for the number of integers or image dimensions, try switching the byte order the other way by changing the '>' to a '<'.

    self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
    self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        
Each of the above lines reads four bytes from the file, where each set of bytes is a 32-bit integer (4 x 8 bits in a byte = 32).  To make sure the number we get is correct, we read them with the reorder_type we just created. Lastly, as we are reading it using a numpy function, it will store the data we read into a numpy array. We want the integer itself rather than an array with one number, so we use [0] to store only the integer.

    buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)
        
Now we are going to read all the image data from the file into a temporary variable that we'll call 'buffer'.  As you know, each byte is a pixel, so we want to read all 60,000 (if it's the training dataset) or 10,000 (if it's the testing dataset) 28x28 pixel images.  To do this, we will use the data we just got about the number of images, row size, and column size to determine the number of bytes we read.

    self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)

This line now reads all the data from the buffer variable as unsigned ints (np.unit8) and stores them into a numpy array as 32-bit floats (np.float32).  We are using floats as Pytorch neural networks, at least the linear ones we'll be using, expect floats rather than double or ints.  This is now our set of image data.  Currently the data is stored as as really long one-dimensional set of numbers.  To see this, you can add print(self.images.shape) like so:

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

As you can see, the range of the sigmoid function where the derivative (slope) changes the most is between 0 and 1.  Thus, learning can quickly occur here.  Having values much larger or smaller than that range can cause neural network saturation, which occurs when the derivatives are so close to 0 that the network can barely learn.  By normalizing our data between 0 and 1, our network will learn much more quickly.

    self.images = torch.tensor(self.images)

Lastly, neural networks work with tensors, not arrays, so we must convert our numpy array of images to Pytorch tensors.  torch.tensor does exactly this, taking numpy array as the argument, converting it to a tensor, and then returning it.

With that, we've finished getting the image data ready for the neural network.  Before we move into getting the label data (which should be much faster), I'll do an optional section on testing.

#### Dataset - Testing Images (Optional)
We're now going to quickly update our draw_image function to be able to actually draw the images we just got. Here's what the code will look like:

    def draw_image(images_root, labels_root, image_idx):
        mnist = MNISTDataset(images_root, labels_root)
        mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
        image = mnist.images[image_idx]
        print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
        plt.imshow(image)
        plt.show()

You'll also want to temporarily comment out the line in the \_\_init\_\_ function that looks like this:

    self.label_init_dataset()

Let me quickly explain the new lines:

    mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
    image = mnist.images[image_idx]
    print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
    plt.imshow(image)
    plt.show()

In the first line we reshape the image set in MNISTDataset object to be 28x28 rather than the flattened 784.  This is so we can see the actual image rather than just a bunch of numbers in a row.  

Next, we directly get the image from the set using the image_idx and then save it to the variable 'image'.  

We then print the dimensions of the image by getting its shape (as shown before when we did it with the image set).  

Finally the last two lines involve drawing the image to the current figure using plt.imshow and then actually displying it with plt.show().  If you want to know more about it, I found [this answer](https://stackoverflow.com/a/3497922/11788566) on Stack Overflow to be quite helpful.

Now run the program and you should be able to see your first digits from MNIST!  Don't forget to uncomment self.label_init_dataset() in the \_\_init\_\_ function as we'll be dealing with that in the next section.

### Dataset - Getting the Label Data
We get the label data in nearly the same way as the image data.  As such, I'll go over thing much more quickly.  Here's the full code:

    def label_init_dataset(self):
        label_file = gzip.open(self.label_data_root, 'r')
        
        reorder_type = np.dtype(np.int32).newbyteorder('>')

        self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        self.num_labels = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        
        buffer = label_file.read(self.num_labels)

        self.labels = np.frombuffer(buffer, dtype = np.uint8)
        
        self.labels = torch.tensor(self.labels, dtype = torch.long)

Section by section:

    def label_init_dataset(self):
        label_file = gzip.open(self.label_data_root, 'r')
        reorder_type = np.dtype(np.int32).newbyteorder('>')

These two lines are nearly the same as the image set.  We open the file and change the byte order.

    self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
    self.num_labels = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]

As I explained way back in the file format, the labels only have two integers in the beginning, the magic number and the number of labels.  Because of this we've only read 8 bytes rather than 16 from the file.

    buffer = label_file.read(self.num_labels)
    self.labels = np.frombuffer(buffer, dtype = np.uint8)
    self.labels = torch.tensor(self.labels, dtype = torch.long)

Then we read all the labels into a variable named 'buffer' using the num_labels value we just got from the initial integers. We then read the labels from the buffer as unsigned ints but now, rather than convert them to floats like we did the image data, we will convert them to longs in the torch.tensor line by making the datatype (dtype) = long.  This time longs are expected for the labels by Pytorch rather than ints.  Keping them as ints would cause an error when we run the neural network later.

With that we are done with our dataset class and can finally move into making the neural network!  Before that though, I want to tweak the draw_image function slightly so you can test that the labels work, and then we can move into making the neural network.

#### Dataset - Testing Images and Labels (Optional)
We're now going to finish up the draw_image function so it works with \_\_getitem\_\_ and allows us to test that our labels work.  No need to comment anything out this time!  Here's the completed function to start with:

    def draw_image(images_root, labels_root, image_idx):
        mnist = MNISTDataset(images_root, labels_root)
        mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
        image, label = mnist.__getitem__(image_idx)
        print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
        print('Label: {}'.format(label.item()))
        plt.imshow(image)
        plt.show()

The first change we're going to switch out how we get the image with the getitem function.

    image, label = mnist.__getitem__(image_idx)

This replaces the 'image = mnist.images[image_idx]' line from the last testing phase.  We are actually using the \_\_getitem\_\_ function here to get the image and label based on the index we give as the argument.  

    print('Label: {}'.format(label.item()))

The last change is adding this little print staement that takes the number from the label tensor (it's currently a tensor with the label in it, but by doing label.item() we can get the number itself) and print it to show that the number matches up with the image.  If the label and image do not match, you may have made a mistake somewhere.

Run this now, and hopefully it'll draw images from the training and testing set along with their corresponding label.  If it works, get ready to move into making the neural network.

### Dataset - Final Code
Here is how the code should look based on what I've given you.  While I generally removed the comments during the earlier parts, I will leave them in for this:

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

            #Functions that intialize the data
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
            #Unzips the image file.  'r' is the mode argument and is just telling it to read the data in binary mode instead of as text
            image_file = gzip.open(self.image_data_root, 'r')
            
            #Datatype that switches the byteorder for the dataset
            reorder_type = np.dtype(np.int32).newbyteorder('>')
            
            #The first 16 bytes of the file are used to give a magic number, the number of images, the rows for each
            #image, as well as the columns for each
            self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
            self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
            self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
            self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]

            #This line reads the rest of the bytes out and puts them in to a temporary variable
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
        #Here, we reshape the the 1D array into a 60000x28x28x1 dimensional array.  This will allow us to be
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
    
## Creating the Neural Network Class
We're now going to create a separate file for the actual neural network class. Create a new file named **NeuralNetwork.py** and we will create this class in there. Surprisingly, the class for the neural network is actually one of the easiest parts of this tutorial.  We only have to create an **\_\_init\_\_** and **forward** function for it. As usual, I'll give you the code first, and then go over it piece by piece:

    import torch.nn as nn

    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            
            self.linear1 = nn.Linear(784, 100)
            self.linear2 = nn.Linear(100, 50)
            self.linear3 = nn.Linear(50, 10)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.linear1(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            x = self.sigmoid(x)
            x = self.linear3(x)
            return x
    
Section by section:

    import torch.nn as nn
    class NeuralNet(nn.Module):

The two easy prerequisites to this are importing the neural network package for Pytorch and then creating the NeuralNet class that extends nn.Module.  Just like how we extended the Dataset class earlier for our dataset, we want to extend the Module class to build our neural network.  All Pytorch neural networks use nn.Module as their base class.  If you're interested, [here](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) is the documentation.

    super(NeuralNet, self).__init__()

Next we use the super function to call the \_\_init\_\_ of the parent function.  The Pytorch documentation says that this is required, however I haven't found the exact reason why this is done.  Usually super is used so you can inhereit methods from the parent, but why it calls the \_\_init\_\_ directly is something I'm not sure of.

    self.linear1 = nn.Linear(784, 100)
    self.linear2 = nn.Linear(100, 50)
    self.linear3 = nn.Linear(50, 10)

Here we create the layers of our neural network to use later in the forward function.  As can be seen in the [documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear), the format for nn.Linear is **nn.Linear(in_features, out_features, bias = True)**.  We are leaving bias as True so we don't need to explicitly change it, but we do need to choose the in and out features.

For linear layer 1, we feed in, as the in_features, all 784 pixels from the image. Then for the out_features I chose a sufficently large number of features I wanted the neural network to look for, in this case 100. 

For the second layer, I took the out_features from the first layer (100) and used them as the in_features here.  I then did the same thing as with the first layer and chose a sufficently large number of features to look for, in this case 50, for the out_features.  You can follow this pattern of 'this layer's out_features become the next layers in_features' to make as large of a neural network as you want.

For the third layer, I used the out_features from the previous layer (50) as the in_features.  As this is the last layer, we want the  out_features for this layer to be the final output/choice of the neural network.  This is the neural network's guess as to which digit between 0-9 the image depicts.  As 0-9 gives us 10 choices, the output will be 10 out_features.  Without any abstraction, the real form of the output will be 10 numbers that increase based off the patterns the neural network has trained for.  The more patterns it sees relating to that digit, the higher the corresponding number in the output.  By choosing the highest of these 10 outputs, we get the neural network's guess.  That part will be done later, though.

    self.sigmoid = nn.Sigmoid()

Lastly we create the sigmoid activation function with [nn.Sigmoid](https://pytorch.org/docs/stable/nn.html#torch.nn.Sigmoid).  We will feed the output of each linear layer into this to create some nonlinearity to the model.  Without this, our linear layers would basically just be doing linear regression.  By adding nonlinear activations, we can better fit our model to the data as our data does not follow a linear pattern.

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        return x

The last function in this class is the forward function, where we build out network with the objects we just created.  Whenever we do a forward pass to send data through the network, this function is used.  Here, x is the input data from the dataset that is passed through as one of the parameters for the function.  We then put this input data through the linear layers and sigmoid activations alternately before returning the output data from the last layer.  This is the structure of one forward pass through the neural network, where the linear layers do a linear transformation on the data and then send the output of the current layer into a nonlinear activation function to define the output in a nonlinear way.  The returned output can then be used to get the guess for the image once the pass ends.

With that, we've finished the neural network.  Now all we have left is the training and testing functions and we're done!

## Creating Training and Testing Program
Our last taks for the tutorial is creating the functions to train and test the neural network.  This is the main program, so we'll call the new file for this **main.py**.  This program will be split into five sections: imports, parameters, training function, testing function, and main.  I will post the full code for the file at the end of those five sections.

### Training and Testing - Imports

    from NeuralNetwork import NeuralNet
    from MNISTDataset import MNISTDataset
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt
    
The imports you'll need for this are the two previous classes we made, NeuralNet and MNISTDataset, as well as torch.nn, torch, and matplotlib.  I think by now these don't really need explanation beyond that we need the first two for loading the dataset into our neural network and we need the last three for playing around with pytorch and plotting values.

### Training and Testing - DataLoader and Parameters
Next we're going to create the dataloaders for each dataset, as well as the parameters for the neural network.  Here's the code:

    training_dataset = MNISTDataset('C:/Easy/train-images-idx3-ubyte.gz', 'C:/Easy/train-labels-idx1-ubyte.gz')
    test_dataset = MNISTDataset('C:/Easy/t10k-images-idx3-ubyte.gz', 'C:/Easy/t10k-labels-idx1-ubyte.gz')

    training_batch_size = 50
    test_batch_size = 1000

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = training_batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

    num_epochs = 25

    neural_net = NeuralNet()
    
    train_losses = []
    train_counter = []
    test_accuracy = []
    test_losses= []

    test_counter = [num * training_dataset.num_images for num in range(num_epochs + 1)]

    loss_function = nn.CrossEntropyLoss()
    
    learning_rate = .2
    momentum = .9
    
    optimizer = torch.optim.SGD(neural_net.parameters(), lr = learning_rate, momentum = momentum)

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

Section by section:

    training_dataset = MNISTDataset('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz')
    test_dataset = MNISTDataset('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz')

Here we create the training and testing dataset objects using the custom dataset class we created before. Once again, make sure the training files go with the training dataset and the testing files go with the testing dataset.

    training_batch_size = 50
    test_batch_size = 1000

Here we will create the batch size for the DataLoader.  Batches are, as the name implies, subsets of the full dataset that are used for training instead of training with the whole dataset at once.  By feeding in batches of samples rather than all of them, it saves on memory usage and makes learning move faster as you aren't calculating the gradient over the whole set. If you want to learn more, [this link](https://datascience.stackexchange.com/a/16818) goes into more detail.  We use batches of 50 for training as you want rather small batches when doing gradient descent and 1000 for testing as it breaks up the 10,000 more evenly.  The batch size doesn't particularly matter for testing so the choice of 1000 was rather arbitrary.  The test size of 50 is slightly abnormal as the standard is usally 32 or 64, but it shouldn't affect the results here in any signficant way.

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = training_batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

These two lines handle loading the two datasets into a [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).  The reason for making the custom Pytorch Datasets is so that we could do this.  The format for the DataLoader object (that we are worried about at least) is **DataLoader(dataset, batch_size, shuffle)**.  The dataset is the custom one we created, the batch_size is what we just defined, and shuffle, when set to True, reshuffles the data every epoch.  Now we can use this dataloader to feed our dataset into the neural network, and it also shuffles the data everytime as a bonus.  Shuffling the dataset serves many purposes, but generally it helps to prevent overfitting as the order which you feed the data into the network may affect the weights of the network if the order remains static across all epochs.  For more complex info on why shuffling is good, look at [this answer](https://datascience.stackexchange.com/a/24539/79065) for the more complex explanation and the one below it for a more simple way of putting it.

    num_epochs = 25

Next we set the number of epochs.  This is simply how long we are training for.  I've found 25 to be a reasonable amount, but feel free to experiment.

    neural_net = NeuralNet()

    train_losses = []
    train_counter = []
    test_accuracy = []
    test_losses= []

Next we intilalize the neural network by creating an object of the neural network class we created in the last section.  This is our actual neural network.  The next four lines are just intializing lists for values we'll be collecting uring training and testing.  We want to be able to append the data to them as the program runs so we can display the loss and accuracy at the end.  

    test_counter = [num * training_dataset.num_images for num in range(num_epochs + 1)]

This is a list of the point for where each epoch ends.  This will later be used to plot the average loss during testing.  What is actually in the test counter variable is basically a list starting with 0 and adding the number of images in the training dataset each times.  So with 60,000 images in the training dataset, this is 0, 60000, 120000, 180000, etc.  The reason we have a 0 here is that we will do an intial test before any training is done to see how well the neural network does with random guessing (no training).

    loss_function = nn.CrossEntropyLoss()

Next we create our loss function using [Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss).  The loss functon is used to evaluate how well the neural network is doing.  Cross entropy loss punishes the model more heavily for being confident in the wrong answer.  We are using it as it is quite good for classification problems.

    learning_rate = .2
    
    momentum = .9

Next we set the learning rate and momentum.  The learning rate affects how quickly the weights of the neural net change.  Too high of a learning rate can skip the optimal weight, well too low of one may cause the model to get stuck in a local minimum.  Due to this, it is worthwhile to test many different learning rate.  It is .2 here as I found it to be of those I tested.  Momentum is used to speed up the convergence of a neural network.  To see a picture of how it works, see [this link](https://www.quora.com/What-exactly-is-momentum-in-machine-learning).

    optimizer = torch.optim.SGD(neural_net.parameters(), lr = learning_rate, momentum = momentum)

Next we create our optimizer.  The optimzer tweaks the weights of the network in order to minimize the loss function.  This makes the model as accuracte as possible.  We are using [Stochaster Gradient Descent](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) here (SGD) which samples a subset of the data in order to determine how to change the weight.  For the arguments, it takes the parameters we want to optimize, the learning rate we just set, and the momentum we also just set.  If you want to know more about what neural_net.parameres() is, check out [this link](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/#parameters-in-depth).

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

The last thing we'll do before we make the training function is to set the seed for randomization so our results will be reproducible.  Basically we choose a random seed here, turn off some backends, and then manually set the seed we chose as the random seed.  Check [here](https://pytorch.org/docs/stable/notes/randomness.html) if you want to know more.

### Training and Testing - Making the Training Function
This next section will cover the creation of the training function.  This function will be actually used to train the neural network.  While small, this code is pretty dense.  As usual, here is the code:

    def train(epoch):
        for batch_idx, (images, labels) in enumerate(training_loader):
            optimizer.zero_grad()

            output = neural_net(images)

            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * training_batch_size, len(training_loader.dataset),
                        100 * batch_idx / len(training_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(training_loader.dataset)))

Section by section:

    def train(epoch):
        for batch_idx, (images, labels) in enumerate(training_loader):

This for loop enumerates over the dataloader.  It gets the batch index and uses the \_\_getitem\_\_ function we made way back in the dataset class to get out the images and labels for each batch.  This for loop will go through all the batches in the training dataset   Now that we have the images and labels, we can now get ready to train the neural network.

















