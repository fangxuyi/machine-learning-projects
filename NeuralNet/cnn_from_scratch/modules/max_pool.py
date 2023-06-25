"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        H_out = int(1 + (H - self.kernel_size) / self.stride)
        W_out = int(1 + (W - self.kernel_size) / self.stride)

        out = np.zeros((N, C, H_out, W_out))
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_offset = h * self.stride
                        w_offset = w * self.stride
                        out[n, c, h, w] = np.max(x[n, c, h_offset: h_offset + self.kernel_size, w_offset: w_offset + self.kernel_size])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_offset = h * self.stride
                        w_offset = w * self.stride
                        current = x[n, c, h_offset: h_offset + self.kernel_size, w_offset: w_offset + self.kernel_size]
                        i, j = np.unravel_index(np.argmax(current, axis=None), current.shape)
                        dx[n, c, h_offset: h_offset + self.kernel_size, w_offset: w_offset + self.kernel_size][i, j] = dout[n, c, h, w]
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
