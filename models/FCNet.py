from . import TorchModel, NUM_GESTURES
import torch
import numpy as np

class FCNet(TorchModel):

    def define_model(self, dim_in):
        dim_in = np.product(dim_in)
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
        predictions = self.model(sample[0].flatten(1).to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), [predictions, None]