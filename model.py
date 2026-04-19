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


def create_model(num_channels=3, num_outputs=12, drop_rate=0.0, learning_rate=0.001, decay=0, pos_weights=None):
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

            self.pool = nn.MaxPool2d(2,2)
            self.conv1 = nn.Conv2d(_num_channels, 16, 5, padding='same') 
            self.conv2 = nn.Conv2d(16, 32, 2, stride=2, padding='valid')
            self.conv3 = nn.Conv2d(32, 64, 2, padding='same')
            self.conv4 = nn.Conv2d(64, 128, 2, stride=2, padding='valid') 
            self.conv5 = nn.Conv2d(128, 256, 2, padding='same')
            self.conv6 = nn.Conv2d(256, 512, 2, padding='same')



            self.fc1 = nn.Linear(2048, 128, bias=False) 
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)

            self.fc2 = nn.Linear(128, 64, bias=False)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)

            self.fc3 = nn.Linear(64, 32, bias=False)
            nn.init.normal_(self.fc3.weight, mean=0.0, std=0.001)

            self.drop = nn.Dropout(p=_drop_rate)

            self.out = nn.Linear(32, _num_outputs, bias=False) 
            nn.init.normal_(self.out.weight, mean=0.0, std=np.sqrt(0.1))

        def forward(self, x):
            """
            Expected input shape: [batch_size, num_channels, H, W]

            Output: tensor w/ shape [batch_size, 12]
            """


            x = F.relu(self.conv1(x)) # out shape: [batch_size, 16, 128, 128]
            x = F.relu(self.conv2(x)) # out shape: [batch_size, 32, 64, 64]
            x = self.pool(x) # out shape: [batch_size, 32, 32, 32]

            x = F.relu(self.conv3) # out shape: [batch_size, 64, 32, 32]
            x = F.relu(self.conv4) # out shape: [batch_size, 128, 16, 16]
            x = self.pool(x) # out shape: [batch_size, 128, 8, 8]

            x = F.relu(self.conv5(x)) # out shape: [batch_size, 256, 8, 8]
            x = self.pool(x) # out shape: [batch_size, 256, 4, 4]

            x = F.relu(self.conv6(x)) # out shape: [batch_size, 512, 4, 4]
            x= self.pool(x) # out shape: [batch_size, 512, 2, 2]

            x = torch.flatten(x) # out shape: [batch_size, 2048,]

            x = F.relu(self.fc1(x))
            x = self.drop(x)

            x = F.relu(self.fc2(x))
            x = self.drop(x)
            
            x = F.relu(self.fc3(x))
            x = self.drop(x)

            x = self.out(x) # No sigmoid activation because this is handled by BCEWithLogitsLoss

            return x #return shape: [batch_size, 12]
        
    model = MultiLabelCNN(_num_channels=num_channels, _num_outputs=num_outputs, _drop_rate=drop_rate)
    model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
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
