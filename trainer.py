import torch
from torch.nn import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models import Generator, Discriminator
from tqdm import tqdm 
from torchvision import datasets
from torchvision.transforms import transforms

def trainer(generator, discriminator, train_dataloader, generator_optimizer, discriminator_optimizer, 
            batch_size, device, epochs=int):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dataloader, desc="Training"):

            discriminator.zero_grad()
            real_images = real_images.to(device)

            real_labels = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device)

            output_real = discriminator(real_images)
            disc_loss_real = criterion(output_real, real_labels)
            disc_loss_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1)
            fake_images = generator(noise)

            output_fake = discriminator(fake_images)
            disc_loss_fake = criterion(fake_labels, output_fake)
            disc_loss_fake.backward()

            total_disc_loss = disc_loss_real + disc_loss_fake

            discriminator_optimizer.step()

            # --------------------------------------------------------- #

            # Train Generator

            generator.zero_grad()

            output_fake = discriminator(fake_images)

            gen_loss = criterion(output_fake, output_fake)

            gen_loss.backward()

            generator_optimizer.step()

            if epoch % 25 == 0:
                print(f'[{epoch}/{epochs}] Loss_D: {total_disc_loss.item():.4f} Loss_G: {gen_loss.item():.4f}')

    if epoch % 10 == 0 or epoch == epochs - 1:
        torch.save(generator.state_dict(), f'./checkpoints/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./checkpoints/discriminator_epoch_{epoch}.pth')

    
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_transform = transforms.Compose(
        transforms.Resize(32, 32),
        transforms.ToTensor(),
    )

    train_data = datasets.CIFAR10(  # Loads the CIFAR-10 training dataset
        root="./data/train",  # Directory where the dataset is stored
        train=True,  # Specifies that we want the training set
        download=False,  # Downloads the dataset if it is not already available
        transform=img_transform  # Applies the defined transformations to the dataset
    )

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    criterion = nn.BCELoss()

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.001)

    trainer(generator, discriminator, train_dataloader, generator_optimizer, discriminator_optimizer, 
            batch_size=32, device=device, epochs=int)