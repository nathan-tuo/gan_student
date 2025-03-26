import torch
import torch.nn as nn


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

    def forward(self, x):

        return
    


if __name__ == "__main__":

    model = Generator()
    noise = torch.randn(32, 100, 1, 1)

    output = model(noise)
    print(output.shape)
    
