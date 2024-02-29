#%% 
import torch, pdb 
from torch.utils.data import DataLoader 
from torch import nn 
from torchvision import transforms 
from torchvision.datasets import MNIST 
from torchvision.utils import make_grid 
from tqdm.auto import tqdm  #progress bar
import matplotlib.pyplot as plt 
# %% visualization function
def show(tensor, ch=1, size=(28,28), num=16):
    # tensor : 128(batch_size) x 784(28*28)를
    # 128 * 1 * 28 * 28 로 transform
    data = tensor.detach().cpu().view(-1, ch, *size) #detach var from the computation of gradients
    grid = make_grid(data[:num], nrow=4).permute(1, 2, 0)    #select 16 pics of the whold data
    # permute(1, 2, 0) means width * height * channel for displaying of matplotlib
    plt.imshow(grid)
    plt.show()
# %% setup of the main parameters and hyperparameters
epochs = 500 
cur_step = 0    #current step
info_step = 300 #every 300 steps to show info 
mean_gen_loss = 0
mean_disc_loss = 0 

#hyperparameters
z_dim = 64    #noise vector
lr = 0.0001     # for ADAM
# logits = the output fo the neural network 
loss_func = nn.BCEWithLogitsLoss()  #apply a sigmoid and calculate BCE

batch_size = 128    #differs from what gpu one uses
device = torch.device('cuda')
# data downloaded on the root dir
dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=batch_size)
# shuffle=True means every epoch the data is newly shuffled
# how many steps are there in each Epoch?
# Total number of images devided by batch_size
# number of steps = MNIST has 60000 data that is to be devided 128 = 468.75
# %% declare our models

# Generator
def genBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=784, h_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim), # 64, 128
            genBlock(h_dim, h_dim*2),
            genBlock(h_dim*2, h_dim*4),
            genBlock(h_dim*4, h_dim*8),
            nn.Linear(h_dim*8, i_dim),   # i_dim = image dimension, that is 28*28
            nn.Sigmoid() # binary classification
        )
    def forward(self, noise):
        return self.gen(noise)

def gen_noise(number, z_dim):
    # randn creates a standard normal distribution
    return torch.randn(number, z_dim).to(device)

# %% Discriminator
def discBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        # LeakyReLU gives out negative values for the value under 0
        nn.LeakyReLU(0.2)    # for the non-linearity
    )

class Discriminator(nn.Module):
    def __init__(self, i_dim=784, h_dim=256):
        super().__init__()
        self.disc = nn.Sequential(
            # h_dim*4 -> starting from the large number of dimension
            discBlock(i_dim, h_dim*4),  # 784, 1024
            discBlock(h_dim*4, h_dim*2),
            discBlock(h_dim*2, h_dim),  # 512, 256
            nn.Linear(h_dim, 1) # only one output, either label 0 or 1
        )
    def forward(self, image):
        return self.disc(image)
# %%
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
# %%
gen
# %%
disc
# %%
x,y = next(iter(dataloader))
print(x.shape, y.shape)
print(y[:10])
# %%
noise = gen_noise(batch_size, z_dim)
fake = gen(noise)
show(fake)  # very noisy because of its pre-training creation
# %% Caculate the loss

# gen loss
def calc_gen_loss(loss_func, gen, disc, number, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    pred = disc(fake) # pass it to discriminator
    targets = torch.ones_like(pred) # label 1 is real one
    gen_loss = loss_func(pred, targets)
    
    return gen_loss

def calc_disc_loss(loss_func, gen, disc, number, real_img, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach()) # detaching so the loss is off of the gradients of the "generator"
    disc_fake_targets = torch.zeros_like(disc_fake) # label 0 is fake one
    disc_fake_loss = loss_func(disc_fake, disc_fake_targets)

    disc_real = disc(real_img)
    disc_real_targets = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_targets)

    disc_loss = (disc_fake_loss + disc_real_loss) / 2  # average loss 

    return disc_loss