"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
not just on MNIST.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable
from . import TorchModel, NUM_GESTURES
import numpy as np

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv1d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        outputs = squash(outputs)
        return outputs

class Structure(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings, conv_filters = 256, conv_kernel_size= 9, conv_stride= 1, conv_padding= 0, primcaps_filters= 256, primcaps_kernel_size= 5, primcaps_stride= 2, primcaps_padding= 0, primcaps_num= 8, digitcaps_dim= 16):
        super(Structure, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 0: Batch Norm
        self.batch_norm1 = torch.nn.BatchNorm1d(np.product(input_size))

        # Layer 1: Just a conventional Conv1D layer
        self.conv = nn.Conv1d(input_size[0], conv_filters, kernel_size=conv_kernel_size,
                              stride=conv_stride, padding=conv_padding)
        self.conv_shape = [conv_filters, int(np.floor(
            (input_size[1] - conv_kernel_size + 2 * conv_padding) / conv_stride) + 1)]

        # Layer 1.0: Batch Norm
        self.batch_norm2 = torch.nn.BatchNorm1d(np.product(self.conv_shape))

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(self.conv_shape[0], primcaps_filters,
                                          dim_caps=primcaps_num,
                                          kernel_size=primcaps_kernel_size,
                                          stride=primcaps_stride, padding=primcaps_padding)
        self.primcaps_shape = [int(np.product([primcaps_filters, int(np.floor(
            (self.conv_shape[1] - primcaps_kernel_size + 2 * primcaps_padding) /
            primcaps_stride) + 1)]) / primcaps_num), primcaps_num]

        # Layer 2.0: Batch Norm
        self.batch_norm3 = torch.nn.BatchNorm1d(np.product(self.primcaps_shape))

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=self.primcaps_shape[0], in_dim_caps=self.primcaps_shape[1],
                                      out_num_caps=classes, out_dim_caps=digitcaps_dim, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(digitcaps_dim * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, np.product(input_size)),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.batch_norm1(x.flatten(1)).view(*x.shape)
        x = self.relu(self.conv(x))
        x = self.batch_norm2(x.flatten(1)).view(*x.shape)
        x = self.primarycaps(x)
        x = self.batch_norm3(x.flatten(1)).view(*x.shape)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        index = length.max(dim=1)[1]
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)

    def loss(self, y_true, y_pred, x, x_recon, lam_recon):
        """
        Capsule loss = Margin loss + lam_recon * reconstruction loss.
        :param y_true: true labels, one-hot coding, size=[batch, classes]
        :param y_pred: predicted labels by CapsNet, size=[batch, classes]
        :param x: input data, size=[batch, channels, width, height]
        :param x_recon: reconstructed data, size is same as `x`
        :param lam_recon: coefficient for reconstruction loss
        :return: Variable contains a scalar loss value.
        """
        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        L_recon = nn.MSELoss()(x_recon, x)

        return L_margin + lam_recon * L_recon

#
# Yet another variation of FullyConnectedNNV2, leveraging the CustomNet module
#
class CapsuleNet(TorchModel):

    def define_model(self, dims):
        model = Structure(dims, NUM_GESTURES, self.capsnet_routings, self.capsnet_conv_filters, self.capsnet_conv_kernel_size, \
                          self.capsnet_conv_stride, self.capsnet_conv_padding, self.capsnet_primcaps_filters,\
                          self.capsnet_primcaps_kernel_size, self.capsnet_primcaps_stride, self.capsnet_primcaps_padding,\
                          self.capsnet_primcaps_num, self.capsnet_digitcaps_dim)
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1].type(torch.LongTensor)).to(self.device)
        if len(targets.shape) == 1:
            targets = torch.zeros(targets.size(0), NUM_GESTURES).to(self.device).scatter_(1, targets.view(-1, 1), 1.)  # change to one-hot coding
        input = sample[0].to(self.device).float()
        predictions, reconstruction = self.model(input)
        return self.model.loss(targets, predictions, input, reconstruction, self.capsnet_lam_recon), [predictions, reconstruction]

    def on_start(self, state):
        super(CapsuleNet, self).on_start(state)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.capsnet_lr_decay)
        """transform = torchvision.transforms.transforms.Compose([
            #torchvision.transforms.transforms.ToPILImage(),
            # torchvision.transforms.transforms.RandomCrop(size=(28, 28), padding=2),
            torchvision.transforms.transforms.ToTensor(),
            # torchvision.transforms.transforms.Normalize((0,), (1,))
        ])
        # dataloaders
        if state['train'] == True:
            #self.train_features = (self.train_features-self.train_features.min()) / (self.train_features.max() - self.train_features.min())
            state['iterator'] = torch.utils.data.DataLoader(
                TorchDatasetIterator(self.train_features, self.train_labels, transform), batch_size=self.batch_size,
                shuffle=True)
            pass
        else:
            state['iterator'] = torch.utils.data.DataLoader(
                TorchDatasetIterator(self.valid_features, self.valid_labels, transform), batch_size=self.batch_size,
                shuffle=True)"""
    def on_end_epoch(self, state):
        self.lr_scheduler.step()
        super(CapsuleNet, self).on_end_epoch(state)