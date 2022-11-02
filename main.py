from secrets import randbits
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


batch_size = 10

def init_process(path, lens, label):
    data = []
    name = label-1
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])
    return data
def Myloader(path):
    return Image.open(path)

class MyData(Dataset):
###########################################################################
# TODO: Complete the custom dataset. You should at least define __init__ ,#
# __getitem__ and __len__  functions                                      #
###########################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def __init__(self, data, transform, loader):
        self.data = data
        self.transfrom = transform
        self.loader = loader
        
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transfrom(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
    


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

class Net(nn.Module):
###########################################################################
# TODO: Complete the neural network. You should at least define __init__  #
# and forward functions                                                   #
###########################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def __init__(self, use_checkpoint=False):
        super(Net, self).__init__()
        self.use_checkpoint=checkpoint
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn18 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*8*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 20)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = F.relu(self.bn6(self.conv6(output)))
        output = self.pool(output)
        output = F.relu(self.bn8(self.conv8(output)))
        output = F.relu(self.bn9(self.conv9(output)))
        output = F.relu(self.bn10(self.conv10(output)))
        output = self.pool(output)
        output = F.relu(self.bn12(self.conv12(output)))
        output = F.relu(self.bn13(self.conv13(output)))
        output = F.relu(self.bn14(self.conv14(output)))
        output = self.pool(output)
        output = F.relu(self.bn16(self.conv16(output)))
        output = F.relu(self.bn17(self.conv17(output)))
        output = F.relu(self.bn18(self.conv18(output)))
        output = output.view(output.shape[0], -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def load_data():

    
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307), std=(0.3081))
    ])
    train_data = []
    test_data = []
    for i in range(1,21):
        if i<10:
            path = 'coil-20-proc/0%d'%(i)+'/obj%d'%(i)+'__%d.png'
            train_data += init_process(path, [0, 60], i)
            test_data += init_process(path, [60, 72], i)
        else:
            path = 'coil-20-proc/%d'%(i)+'/obj%d'%(i)+'__%d.png'
            train_data += init_process(path, [0, 60], i)
            test_data += init_process(path, [60, 72], i)

    train = MyData(train_data, transform=transform, loader=Myloader)
    test = MyData(test_data, transform=transform, loader=Myloader)
    train_data = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data, test_data

Cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
train_data, test_data = load_data()
model = Net(use_checkpoint=False).cuda()
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def saveModel():
    path = "./Model.pth"
    torch.save(model.state_dict(), path)

#Function to test the model with the test dataset and print the accuracy for the test image
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    for data in test_data:
        images, labels = data
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        accuracy += (predicted == labels).sum().item()
    accuracy = (100 * accuracy) / total
    return accuracy

def train(num_epochs):

    best_accuracy = 0.0
    best_loss = 100.0
    count = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (images, labels) in enumerate(train_data, 0):
            
            #get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.detach().item()

        print('For epoch %d loss: %.3f' % (epoch+1, running_loss / 1000))
        # zero the loss
        running_loss = 0.0
        if(running_loss < best_loss):
            best_loss = running_loss

        count += 1
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        if(accuracy > best_accuracy):
            saveModel()
            best_accuracy = accuracy

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_data))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%d' % labels[j] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%d' % predicted[j] 
                              for j in range(batch_size)))



if __name__ == "__main__":
###############################################################################
# TODO: Complete the train and test process. You should split train and test  #
# dataset by a ratio of 5:1. The accuracy on the test set should be above 95%.#
###############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #Cuda = True
    train(20)
    print('Finished Training')
    #testModelAccuracy()


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****