import torch
import torch.nn as nn
from torchinfo import summary


# Feeding noise to Generator... input: (32, 100, 1, 1) output: (32, 3, 32, 32)
class Generator(nn.Module):
    def __init__(self, d=128, out_features=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100, d*8, 4, 1, 0), # ([32, 1024, 4, 4])
            nn.BatchNorm2d(d*8), 
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1), # ([32, 512, 8, 8])
            nn.BatchNorm2d(d*4),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1), #([32, 256, 32, 32])
            nn.BatchNorm2d(d*2), 
            nn.ConvTranspose2d(d*2, out_features, 4, 2, 1), # ([32, 3, 32, 32])
        )

    def forward(self, x):
        
        return self.layers(x)



class Discriminator(nn.Module):
    def __init__(self, out_features=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),  # Activation function

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        return self.sigmoid(x)


if __name__ == "__main__":

    gen = Generator()
    noise = torch.randn(32, 100, 1, 1)

    output = gen(noise)
    print(output.shape)


    disc = Discriminator()
    image = torch.randn(32, 3, 32, 32)

    output = disc(image)
    print(output.shape)

    summary(gen, input_size=(noise.shape))
    summary(disc, input_size=(image.shape))
