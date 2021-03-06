'''
Predicts which characters are fighting in the scene
'''
import os
import torch
import torch.nn as nn

from landmarks.constants import (
    LABEL_TO_ID,
    MODEL_DIR,
)


class LeNetLayer(nn.Module):
    '''
    Simple LeNet layer
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 pool_dims=None):
        '''
        Create a single LeNet layer
        '''
        super(LeNetLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride)

        self.pool_dims = pool_dims

    def forward(self, x):
        '''
        Feed forward through one LeNet layer
        '''
        x = self.conv(x)
        x = torch.relu(x)

        if self.pool_dims:
            x = torch.max_pool2d(x, self.pool_dims)

        return x


class LandmarkModel(nn.Module):
    '''
    Net that learns to classify an Arabic letter from an image
    '''
    def __init__(self, load_weights=True):
        '''
        Creating the network
        '''
        super(LandmarkModel, self).__init__()
        self.layer1 = LeNetLayer(1, 32, pool_dims=2)
        self.layer2 = LeNetLayer(32, 64, pool_dims=2)
        self.layer3 = LeNetLayer(64, 64, pool_dims=2)
        self.layer4 = LeNetLayer(64, 64, pool_dims=2)
        self.layer5 = LeNetLayer(64, 128)

        # this is our final classification layer
        self.fc1 = nn.Linear(512, len(list(LABEL_TO_ID.keys())))

        if load_weights:
            self.load_weights()

    def forward(self, x):
        '''
        Forward prop
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x

    def load_weights(self):
        '''
        Loads the weights for the model
        '''
        print('Loading the weights for CharacterPredictModel')
        weights = torch.load(os.path.join(
            MODEL_DIR, '2_fixed_softmax_sigmoid_error__400'))
        self.load_state_dict(weights)

    def predict(self, image):
        '''
        Predict on an image.  Returns the class_id and the confidence
        '''
        self.eval()

        with torch.no_grad():
            output = self(torch.unsqueeze(image, 0))
            return output
