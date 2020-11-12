import time
from typing import List
import functools
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import h5py


from .neural_model import NeuralModel as Model

# Hyperparameters
NUM_DOWNS = 8
NGF = 64
LAMBDA_L1 = 10

class Dataset(data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]


# Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
class Pix2PixUnet(Model, nn.Module):
    """Create a Unet-based generator"""

    def __init__(self):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()

        # hyper params
        self.num_downs = NUM_DOWNS
        self.ngf = NGF
        self.lambda_L1 = LAMBDA_L1
        self.norm_layer = nn.BatchNorm2d

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(self.ngf * 8, self.ngf * 8, input_nc=None, submodule=None, norm_layer=self.norm_layer, innermost=True)  # add the innermost layer
        for i in range(self.num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(self.ngf * 8, self.ngf * 8, input_nc=None, submodule=unet_block, norm_layer=self.norm_layer, use_dropout=False)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(self.ngf * 4, self.ngf * 8, input_nc=None, submodule=unet_block, norm_layer=self.norm_layer)
        unet_block = UnetSkipConnectionBlock(self.ngf * 2, self.ngf * 4, input_nc=None, submodule=unet_block, norm_layer=self.norm_layer)
        unet_block = UnetSkipConnectionBlock(self.ngf, self.ngf * 2, input_nc=None, submodule=unet_block, norm_layer=self.norm_layer)
        self.model = UnetSkipConnectionBlock(self.output_c, self.ngf, input_nc=self.input_c, submodule=unet_block, outermost=True, norm_layer=self.norm_layer)  # add the outermost layer

        self.device = torch.device(
            f"cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {self.device}")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterionL1 = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)


    def train(self):
        # load data from file
        with h5py.File('C:/Users/Kai/GitHub/Forever_Machine_Learning/code/data_950.h5', 'r') as hf:
            data_x = hf['x'][:]
            data_y = hf['y'][:]

        # 80/20 split
        # data_tr, data_te = data_obs.split()
        train_ind = int(len(data_x) * 0.8)
        data_tr = Dataset(data_x[:train_ind], data_y[:train_ind])
        data_te = Dataset(data_x[train_ind:], data_y[train_ind:])

        train_loader = DataLoader(
            data_tr,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        train_losses = []

        for epoch in range(self.num_epochs):
            start_time1 = time.time()

            epoch_loss = torch.tensor(0.0, device=self.device)

            for batch_idx, batch in enumerate(train_loader):
                start_time2 = time.time()

                data, target = batch

                outputs = self.model(data.float().to(self.device))

                loss_gen = self.criterion(outputs, target.float().to(self.device))
                loss_L1 = self.criterionL1(outputs, target) * self.lambda_L1
                loss = loss_gen + loss_L1

                # Aggregate loss across mini-batches (per epoch)
                epoch_loss += loss

                # Backprop and perform Adam optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(
                f"Epoch: {epoch}\tBatch: {batch_idx}\tGen Loss: {loss_gen:.5f}\tL1 Loss: {loss_L1:.5f}"
                f"\tTotal Time: {(time.time() - start_time2)/60:.3f} minutes"
                )

            print(
                f"Epoch: {epoch}\tTrain Loss: {epoch_loss/len(train_loader):.5f}"
                f"\tTotal Time: {(time.time() - start_time1)/60:.3f} minutes"
            )

            train_losses.append(epoch_loss / len(train_loader))

        return train_losses



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
