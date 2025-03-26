import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models import Generator, Discriminator
from tqdm import tqdm 

def trainer(generator, train_dataloader, discriminator, gen_optim, disc_optim, epochs=int):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for _, _ in tqdm(train_dataloader):



    return 
    




if __name__ == "__main__":

    train_dataloader = DataLoader()

    generator_optimizer = Adam(Generator.parameters(), lr=0.001)
    discriminator_optimizer = Adam(Discriminator.parameters(), lr=0.001)