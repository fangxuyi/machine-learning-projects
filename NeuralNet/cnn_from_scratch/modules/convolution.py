"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape
        H_prime = int(1 + (H + 2 * self.padding - self.kernel_size) / self.stride)
        W_prime = int(1 + (W + 2 * self.padding - self.kernel_size) / self.stride)
        padded_x = np.pad(x, [(0,), (0,), (self.padding,), (self.padding,)], 'constant', constant_values=0)

        out = np.zeros((N, self.out_channels, H_prime, W_prime))
        for n in range(N):
            for o in range(self.out_channels):
                for h in range(H_prime):
                    for wi in range(W_prime):
                        h_offset = h * self.stride
                        w_offset = wi * self.stride
                        out[n, o, h, wi] = np.sum(padded_x[n, :, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size] * self.weight[o]) + self.bias[o]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        _, _, H_prime, W_prime = dout.shape

        dx = np.zeros(x.shape)
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    h_idxs = []
                    w_idxs = []
                    for i in range(H_prime):
                        for j in range(W_prime):
                            i_offset = i * self.stride
                            j_offset = j * self.stride
                            if ((h + self.padding - i_offset) >= 0) and ((h + self.padding - i_offset) < self.kernel_size) and \
                                    ((w + self.padding - j_offset) >= 0) and ((w + self.padding - j_offset) < self.kernel_size):
                                h_idxs.append((i, j))
                                w_idxs.append((h + self.padding - i_offset, w + self.padding - j_offset))

                    for o in range(self.out_channels):
                        dx[n, :, h, w] += np.sum([self.weight[o, :, widx[0], widx[1]] * dout[n, o, hidx[0], hidx[1]] for hidx, widx in zip(h_idxs, w_idxs)], 0)

        dw = np.zeros((self.out_channels, C, self.kernel_size, self.kernel_size))
        padded_x = np.pad(x, [(0,), (0,), (self.padding,), (self.padding,)], 'constant', constant_values=0)
        for o in range(self.out_channels):
            for c in range(C):
                for h1 in range(self.kernel_size):
                    for w1 in range(self.kernel_size):
                        h1_offset = h1 * self.stride
                        w1_offset = w1 * self.stride
                        dw[o, c, h1, w1] = np.sum(padded_x[:, c, h1_offset: h1_offset + H_prime, w1_offset: w1_offset + W_prime] * dout[:, o, :, :])

        self.db = np.sum(dout, axis=(0, 2, 3))
        self.dw = dw
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
