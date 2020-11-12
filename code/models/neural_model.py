from typing import List
import torch.nn as nn
import torch

# Hyperparameters (Put here for now. Probably should find a way to organize them)
SIZE = 256
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
NUM_EPOCHS = 20
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
SHUFFLE = True


class NeuralModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load in default parameters
        self.size = SIZE
        self.input_c = INPUT_CHANNELS
        self.output_c = OUTPUT_CHANNELS
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE
        self.shuffle = SHUFFLE


    def forward(self, obs):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def prepare_batch(self, data):
        pass


