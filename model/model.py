import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class Net(nn.Module):
    def __init__(self, out_classes):
        super(Net, self).__init__()
        # convolutional layer (sees 224x224x3 image tensor)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)

        # convolutional layer (sees 112x112x8 tensor)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        # convolutional layer (sees 56x56x16 tensor)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)

        # convolutional layer (sees 28x28x32 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        # convolutional layer (sees 14x14x64 tensor)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

        # above conv4 outputs after max pooling 7x7x64

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (7 * 7 * 64 -> 1000)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)

        # linear layer (1000 -> 500)
        self.fc2 = nn.Linear(1000, 500)

        # linear layer (500 -> 4)
        self.fc3 = nn.Linear(500, out_classes)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # flatten image input
        _, d, h, w = x.shape
        x = x.view(-1, 7 * 7 * 64)

        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)

        # checking with softmax function
        # x = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x
