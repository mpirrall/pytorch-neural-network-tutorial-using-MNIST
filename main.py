#A good portion of this class was modified from code in Gregor Koehler's article here: https://nextjournal.com/gkoehler/pytorch-mnist

from NeuralNetwork import NeuralNet
from MNISTDataset import MNISTDataset
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


#Creates the MNIST Dataset objects for the test and training datasets from their respective files
training_dataset = MNISTDataset('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz')
test_dataset = MNISTDataset('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz')

#These are the training and testing batch sizes
training_batch_size = 50
test_batch_size = 1000

#Loads the datasets into dataloaders so the datasets can be enumerated
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = training_batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

#Variable for the number of epochs we are training for
num_epochs = 25

#Creates the neural network object
neural_net = NeuralNet()

#Creating empty lists to append values to during training and testing
train_losses = []
train_counter = []
test_accuracy = []
test_losses= []

#This creates a list with the point for where each epoch ends.  This will later be used to plot the average loss
#during testing  It's num_epochs + 1 because it will also make a point for when no training has been done and 
#the weights are just random.  You would remove the +1 if you did not do an initial test
test_counter = [num * training_dataset.num_images for num in range(num_epochs + 1)]

#The loss function is created. The loss function evaluates how well the neural network is doing
#We use Cross Entropy Loss as it punishes the model more heavily for being confident in a wrong answer.  
loss_function = nn.CrossEntropyLoss()
#The learning rate is how quickly the weights of the neural net change.  Too high of a learning rate may make
#the model skip the optimal value, while too low may make the model get stuck in a local minimum
learning_rate = .2
#Momentum in stochastic gradient descent
momentum = .9
#The optimizer tweaks the weights of the network in order to minimize the loss function.  This makes the 
#model as accurate as possible.  Here we are using Stochastic Gradient Descent, which samples a subset
#of the data to determine how to change the weights.
#neural_net.parameters() - Look for the "Parameters In-Depth" heading here: 
#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/#parameters-in-depth
optimizer = torch.optim.SGD(neural_net.parameters(), lr = learning_rate, momentum = momentum)

#Sets the seed for randomization manually so that results can be reproducible during testing
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


#This is the training function.
def train(epoch):
    #For loop that enumerates over the training loader.  It has a batch index and gets out the images and labels
    for batch_idx, (images, labels) in enumerate(training_loader):
        #If you do not call this, the gradients will accumulate over time.  As gradients are computed during
        #loss.backward() but the gradients themselves are not used to proceed gradient descent until
        #optimizer.step(), we have to manually zero the gradients at the start of the loop.
        optimizer.zero_grad()
        
        #Here we put the batch of images into the neural net and get an output. The output is basically the guesses
        #of the model for each image.  This is where we are actually using the model. The output is a tensor of 
        #Size([50, 10]).  The 50 is the row for each image in the batch and the 10 is the values for the model's 
        #confidence in the image being each digit.  For a more in depth explanation, look a the comment above 
        #guesses in the test function
        output = neural_net(images)
        
        #Here we compute the loss from the output and labels.  We compare what answers we got from the model
        #for the images in this batch (output) to the actual answers for the batch (labels).  The loss function
        #takes this info and uses it to compute the loss
        loss = loss_function(output, labels)
        
        #loss.backward() calculates the gradients for every parameter in your model with requires_grad = True
        #Once the gradients are accumulated, they can then be used by the optimizer to update the parameters
        loss.backward()
        #optimizer.step() is where the values of the parameters are updated using the gradients computed in
        #loss.backward().  This is more or less where the training occurs
        optimizer.step()
        
        #This occurs if the batch index modulus 20 = 0.  So if the batch size was 50, and this occurs every 20 batches,
        #then this occurs for every 50*20 samples, or every 1000 samples.  In short, the number you choose
        #for the modulus is every how many batches this prints
        if batch_idx % 100 == 0:
            #This print statement just shows the epoch, how far you are, and the current loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * training_batch_size, len(training_dataset),
                    100 * batch_idx / len(training_loader), loss.item()))
            #This line appends the current loss to a list so it can be graphed later
            train_losses.append(loss.item())
            #This adds the index of the current training sample by finding where it is in the current epoch which
            #is batch_idx * training_batch_size, plus how all the indexes that have passed from previous epochs, which
            #is (epoch-1) * len(training_loader.dataset)
            train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(training_dataset)))
            
            #These lines save the neural net and optimizer so they can be accessed and updated later
            #torch.save(neural_net.state_dict(), 'C:/results/model.pth')
            #torch.save(optimizer.state_dict(), 'C:/results/optimizer.pth')

def test():
    test_loss = 0
    correct_guesses = 0
    #This wrapper will temporarily set all requires_grad flags to false.  As we know we will not be computing 
    #gradients in our testing function, this will help speed it up and reduce memory usage.  Use torch.no_grad()
    #when you do not need to compute gradients or backpropagate.  Almost always should be used in testing functions
    with torch.no_grad():
        #for loop that gets the images and labels from the test loader.  Images are in batches of 1000 as set during
        #the initialization of the dataloader
        for images, labels in test_loader:
            #We get the model's guesses for what the images in the batch are
            output = neural_net(images)
            #This adds up the loss for each batch of the test run
            test_loss += loss_function(output, labels).item()
            #As the output is a test_batch_sizex10 (1000x10 if this code is unmodified) tensor, each image is 
            #a 1x10 tensor with each of the 10 values corresponding to the model's confidence in the image being
            #the value's respective number from 0-9.  Ex: the value of index 0 of the 1x10 tensor corresponds to the 
            #confidence that the image is 0, while the value of index 1 corresponds to the confidence in it being 1, etc.
            #Thus, the index of the number with the highest confidence (maximum value) is the model's guess, which we want.
            #The return type of torch.max is a tuple of (values, indices) where values is the max value of each row
            #and indices is the index of each max value.  By getting [1] of this tuple, we are getting the indices of
            #the max values, which correspond to the guess of the digits, as explained in the example above.
            guesses = torch.max(output, 1, keepdim = True)[1]
            #torch.eq computes element-wise equality for two tensors, meaning that it checks each element in the 
            #input tensor (guesses) againt it's respective element in the other tensor (labels).  It then outputs a 
            #ByteTensor with a 1 at each location where the comparisons are true.  By taking the sum of all the 1's
            #in the ByteTensor, we can get the amount of correct guesses the model made.
            #The view_as function works by viewing the tensor as the same size as the input tensor.  So doing
            #labels.data.view_as(guesses) means that we are going to the labels tensor (of Size([1000])), and 
            #view it as a tensor or the same size as guesses (Size([1000, 1]).  This makes the two tensors
            #'broadcastable' in pytorch, which allows us to do operations on them
            correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum()
        #The test_loss was summed throughout the test, and is now averaged by dividing it by the length of the dataset
        #over the size of the test batches.  In simpler terms, this is the combined loss of all batches divided by
        #the total number of batches
        test_loss /= len(test_loader.dataset)/test_batch_size
        #The average loss for this test run is then appended to a list of all the test losses.  This will be used
        #when graphing the results
        test_losses.append(test_loss)
        #Calculates the accuracy for the test run and append it to the list
        current_accuracy = float(correct_guesses)/float(len(test_dataset))
        test_accuracy.append(current_accuracy)
        #Print statement to show the average loss and accuracy
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct_guesses, len(test_dataset),
                100. * current_accuracy))

if __name__ == "__main__":
    #This does an initial test of the model when all the weights are random.  This is the initial test mentioned in the
    #comment for test_counter
    test()
    #This for loop tests for the number of epochs we want.  As it starts at 1, we do the number of epochs + 1
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test()
    
    #This line just prints the number of epochs we trained for in total
    print('Total epochs: {}'.format(num_epochs))
    #This line prints the maximum accuracy.  We find the maximum test accuracy (which is in the form 0.97355... or 0.96242...)
    #and multiply it by 100 (so it is in the form 97.355... or 96.242...).  Then we round it to the second decimal place, giving
    #us a percentage of the form 97.35 or 96.24.
    print('Max Accuracy is: {}%'.format(round(100*max(test_accuracy), 2)))
    
    #Below is the set of functions to graph the results
    fig = plt.figure()
    #This plots the loss during the training in blue
    plt.plot(train_counter, train_losses, color = 'blue', zorder = 1)
    #This plots the loss during testing in red
    plt.scatter(test_counter, test_losses, color = 'red', zorder = 2)
    
    plt.scatter(test_counter, test_accuracy, color = 'green', marker = '+', zorder = 3)
    plt.legend(['Train Loss', 'Test Loss', 'Accuracy'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
