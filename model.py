"""

The model code itself

Hyperparams: drop_rate, lr, weight_decay (regularization)

model expects tensor of shape [BATCH_SIZE, Channels, H, W]. 

output of model.forward: tensor of length NUM_OUTPUTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.models.resnet import Bottleneck, BasicBlock # Taking bottleneck layer from ResNet to use in our model

def create_model(num_channels=3, num_labels=12, drop_rate=0.0, learning_rate=0.001, decay=0, pos_weights=None):
    """
    Function to create the model. Attaches an Adam optimizer and uses BCEWithLogitsLoss for loss function.

    Hyperparameters:
    - drop_rate: probability of dropout for the drop out layer. Default is no dropout.
    - learning_rate: learning rate of the optimizer
    - decay: L2 regularization on the optimizer. Default is no regularization.
    - pos_weights: Allows loss function to account for class imbalance


    Output:
    Model object containing model, optimizer and loss function.

    """

    class MultiLabelCNN(nn.Module):
        # Current parameter count: 973344
        def __init__(self, _num_channels=3, _num_outputs = 12, _drop_rate=0.0): 
            super(MultiLabelCNN, self).__init__()
            """
            self.res = resnet50(weights=None) # pull ResNet 50 as backbone, but with empty weights
            if fetch_weights:
                state_dict = torch.load('resnet50.pth') # load pretrained file
                self.res.load_state_dict(state_dict)
            self.backbone = nn.Sequential(*list(self.res.children())[:-2]) # only go to layer 4
            for param in self.res.parameters(): # freeze parameters
                param.requires_grad = False
            """


            self.pool = nn.MaxPool2d(2,2)
            self.conv1 = nn.Conv2d(_num_channels, 32, 5, padding='same') 
            self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
            self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
            self.conv4 = nn.Conv2d(128, 512, 3, padding='same') 
            self.conv5 = nn.Conv2d(512, 512, 3, padding='same')
            self.conv6 = nn.Conv2d(512, 512, 3, padding='same')
            self.conv7 = nn.Conv2d(512, 512, 3, padding='same')
            self.conv8 = nn.Conv2d(512, 512, 3, padding='same')

            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(512)
            self.bn6 = nn.BatchNorm2d(512)
            self.bn7 = nn.BatchNorm2d(512)
            self.bn8 = nn.BatchNorm2d(512)

            # layers for matching residual to x when doing skip connections
            self.downsample1 = nn.Conv2d(3, 64, 1, stride=4)
            self.downsample2 = nn.Conv2d(64, 512, 1, stride=4)
            self.downsample3 = nn.Conv2d(512, 512, 1, stride=4)
            self.downsample4 = nn.Conv2d(512, 512, 1, stride=2)
            #self.bottle = Bottleneck(256, 64)

            #self.fc1 = nn.Linear(2048, 512, bias=True) 
            #nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)
            #nn.init.zeros_(self.fc1.bias)

            #self.fc2 = nn.Linear(512, 128, bias=True)
            #nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)
            #nn.init.zeros_(self.fc2.bias)

            #self.fc3 = nn.Linear(128, 32, bias=False)
            #nn.init.normal_(self.fc3.weight, mean=0.0, std=0.001)

            self.drop = nn.Dropout(p=_drop_rate)

            self.out = nn.Linear(512, _num_outputs, bias=False) 
            nn.init.xavier_uniform_(self.out.weight) # intended for sigmoid activation


        def forward(self, x):
            """
            Expected input shape: [batch_size, num_channels, H, W]

            Output: tensor w/ shape [batch_size, 12]
            """

            identity = x # implement skip connections
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)

            x += self.downsample1(identity)
            identity = x

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)

            x += self.downsample2(identity)
            identity = x

            x = F.relu(self.bn5(self.conv5(x))) 
            x = self.pool(x) 
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool(x)

            x += self.downsample3(identity)
            identity = x

            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            x = self.pool(x)

            x += self.downsample4(identity)

            x = torch.flatten(x,1)
            x = self.drop(x)
            x = self.out(x) # No sigmoid activation because this is handled by BCEWithLogitsLoss

            return x #return shape: [batch_size, 12]
        
    model = MultiLabelCNN(_num_channels=num_channels, _num_outputs=num_labels, _drop_rate=drop_rate)
    model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    #model.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)
    model.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    return model


   
def main():
    # Test that the model can actually be initialized and get param count
    in_shape = (256, 3, 128, 128)

    model = create_model(3, 12)
    print(f"Model initialized with:")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{params} params")
    print(f"Optimizer: {type(model.optimizer)}")
    print(f"Loss func: {type(model.loss_func)}")
    print(model)
    


if __name__=="__main__":
    main()
