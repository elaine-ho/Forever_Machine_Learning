import torch

from models.cnn import ConvolutionalNeuralNetwork as CNN
from models.gan import GenerativeAdversarialNetwork as GAN
from helpers.neural_helpers import print_model


if __name__ == '__main__':

    # hyper parameters
    obs_size = 5
    filters = [4, 8]
    flatten = [16]
    kernel = 3

    cnn = CNN((2, obs_size, obs_size), 4, filters, flatten, kernel)
   
    X = torch.randn(size=(1, 2, 5, 5), dtype=torch.float32)
    print_model(cnn.model, X)



    # cnn.train(filters, flatten, kernel)


