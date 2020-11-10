import time
from typing import List
import torch.nn as nn
import torch

from .neural_model import NeuralModel as Model

class ConvolutionalNeuralNetwork(Model, nn.Module):
    # def __init__(self, params):  # parameters need to be consolidated.
    # Maybe pass parameters as a yaml file and have a helper parameter file parse it.    
    def __init__(self, obs_size, action_size, filters, flatten, kernel):
        super().__init__()

        self.name = "cnn"

        self.filters = [obs_size[0]] + filters
        self.flatten = [filters[-1]*obs_size[-2]*obs_size[-1]] + flatten
        self.kernel = kernel
        self.padding = (kernel - 1) // 2

        layers: List[nn.Module] = []

        # conv layers
        for i in range(1, len(self.filters)):
            layers.append(
                nn.Conv2d(
                    self.filters[i - 1],
                    self.filters[i],
                    self.kernel,
                    padding=self.padding,
                )
            )
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())

        # dense layers
        for i in range(1, len(self.flatten)):
            layers.append(
                nn.Linear(
                    self.flatten[i - 1],
                    self.flatten[i]
                )
            )
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.flatten[-1], action_size))

        self.model = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()


    def forward(self, x):
        return self.model(x)


    def train(self):
        #TODO: Call a load data function. Do the 80/20 split.
        # Use torch.utils.data.DataLoader to load data.
        train_loader = []  #TODO: replace with training data

        train_losses = []

        for epoch in range(self.num_epochs):
            start_time = time.time()

            epoch_loss = torch.tensor(0.0)

            for batch_idx, batch in enumerate(train_loader):

                #TODO: Call a function to prepare batch of data.
                data, target = [], []  #TODO: replace with loaded data

                outputs = self.model(data)

                loss = self.criterion(outputs, target)
                
                # Aggregate loss across mini-batches (per epoch)
                epoch_loss += loss

                # Backprop and perform Adam optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch: {epoch}\tTrain Loss: {epoch_loss/len(train_loader):.5f}"
                f"\tTotal Time: {time.time() - start_time:.3f}"
            )

            train_losses.append(epoch_loss / len(train_loader))

