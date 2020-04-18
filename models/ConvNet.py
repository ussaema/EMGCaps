from . import TorchModel, NUM_GESTURES
import torch
from torch import nn
import numpy as np

class ConvNet(TorchModel):

    def define_model(self, dim_in):

        self.conv = nn.Conv1d(dim_in[0], self.conv_filters, kernel_size=self.conv_kernel_size,
                              stride=self.conv_stride, padding=self.conv_padding)
        self.conv_shape = [self.conv_filters, int(np.floor(
            (dim_in[1] - self.conv_kernel_size + 2 * self.conv_padding) / self.conv_stride) + 1)]

        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.Linear(dim_in, dim_in * 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 32, dim_in * 64),
            torch.nn.BatchNorm1d(dim_in * 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 64, NUM_GESTURES),
            torch.nn.Softmax(dim=1)
        )
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1].type(torch.LongTensor)).to(self.device)
        predictions = self.model(sample[0].to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), [predictions, None]

from . import TorchModel, NUM_GESTURES
import torch

class Structure(torch.nn.Module):
    """
        The classifier with the best known performance on the NinaPro dataset thus far (using a variation of
            PaddedMultiRMS).
    """

    def __init__(self, input_size, classes, convnet_filters, convnet_kernel_size, convnet_stride, convnet_padding, convnet_maxpooling, convnet_fc_num):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Structure, self).__init__()
        # Layer 0: Batch Norm
        self.batch_norm1 = torch.nn.BatchNorm1d(np.product(input_size))

        # Layer 1: Conv Layer
        self.conv = nn.Conv1d(input_size[0], convnet_filters, kernel_size=convnet_kernel_size,
                              stride=convnet_stride, padding=convnet_padding)
        self.conv_shape = [convnet_filters, int(np.floor(
            (input_size[1] - convnet_kernel_size + 2 * convnet_padding) / convnet_stride) + 1)]

        # Layer 1.0: Maxpooling Layer
        self.maxpool = nn.MaxPool1d(convnet_maxpooling)
        self.maxpool_shape = [self.conv_shape[0], self.conv_shape[1] // convnet_maxpooling]

        # Layer 2: FC Layer
        self.fcn = nn.Sequential(
            nn.Linear(np.product(self.maxpool_shape), convnet_fc_num),
            nn.ReLU(inplace=True),
            nn.Linear(convnet_fc_num, classes),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(x.flatten(1)).view(*x.shape)
        x = self.relu(self.conv(x))
        x = self.maxpool(x)
        y = self.fcn(x.flatten(1))
        return y

#
# Yet another variation of FullyConnectedNNV2, leveraging the CustomNet module
#
class ConvNet(TorchModel):
     def define_model(self, dim_in):
         model = Structure(dim_in, NUM_GESTURES, self.convnet_filters, self.convnet_kernel_size, self.convnet_stride, self.convnet_padding, self.convnet_maxpooling, self.convnet_fc_num)
         return model

     def forward_pass(self, sample):
         targets     = torch.LongTensor(sample[1].type(torch.LongTensor)).to(self.device)
         predictions = self.model(sample[0].to(self.device))
         return torch.nn.functional.cross_entropy(predictions, targets), [predictions, None]
