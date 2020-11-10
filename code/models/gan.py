import time
from typing import List
import torch.nn as nn
import torch

from .neural_model import NeuralModel as Model

class Generator(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        pass
        
    def forward(self, x):
        return self.model(x)



class GenerativeAdversarialNetwork(Model, nn.Module):
    # def __init__(self, params):  # parameters need to be consolidated.
    # Maybe pass parameters as a yaml file and have a helper parameter file parse it.
    def __init__(self, obs_size, action_size, filters, flatten, kernel):
        super().__init__()

        #TODO: Initialize necessary params for both models

        self.name = "gan"

        self.generator = Generator()  #TODO: pass necessary params here
        self.discriminator = Discriminator()  #TODO: pass necessary params here

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.criterion = nn.BCELoss()


    def forward(self, obs):
        return self.model(obs)


    # implementations that I looked at:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    https://machinelearningmastery.com/how-to-code-the-generative-adversarial-network-training-algorithm-and-loss-functions/
    def train(self):
        #TODO: Call a load data function. Do the 80/20 split. 
        # Use torch.utils.data.DataLoader to load data.
        train_loader = []  #TODO: replace with training data

        train_g_losses = []
        train_d_losses = []

        for epoch in range(self.num_epochs):
            start_time = time.time()

            epoch_g_loss = torch.tensor(0.0)
            epoch_d_loss = torch.tensor(0.0)

            for batch_idx, batch in enumerate(train_loader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                #### Train Generator ####
                self.optimizerG.zero_grad()

                #TODO: Call a function to prepare batch of data.
                real_data = []  #TODO: replace with loaded data

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # generate fake images
                gen_data = self.generator(z)

                g_loss = self.criterion(self.discriminator(gen_data), valid)
                
                g_loss.backward()
                self.optimizerG.step()


                 #### Train Discriminator ####
                self.optimizerD.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.criterion(self.discriminator(real_data), valid)
                fake_loss = self.criterion(self.discriminator(gen_data.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizerD.step()

            print(
                f"Epoch: {epoch}\tTrain Loss G: {epoch_g_loss.data/len(train_loader):.5f}\tTrain Loss D: {train_d_loss.data/len(train_loader):.5f}"
                f"\tTotal Time: {time.time() - start_time:.3f}"
            )

            train_g_losses.append(epoch_g_loss / len(train_loader))
            train_d_losses.append(epoch_d_loss / len(train_loader))


