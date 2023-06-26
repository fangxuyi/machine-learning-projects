"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        #vgg block
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=0) # 32 - 3 + 1 = 30
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)  # 30 - 3 + 1 = 28
        self.pool1 = nn.MaxPool2d(2, stride=2) # (28 - 2) / 2 + 1 = 14
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # 14 - 3 + 1 = 12
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  # 12 - 3 + 1 = 10
        self.pool2 = nn.MaxPool2d(2, stride=2) # (10 - 2) / 2 + 1 = 5
        self.fc = nn.Linear(256 * 5 * 5, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(nn.functional.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        outs = nn.functional.relu(self.fc(x))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
