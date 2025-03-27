import torch
import torch.nn as nn
from torchinfo import summary
from huggingface_hub import PyTorchModelHubMixin

# Conditional Generator - now takes class label as input
class ConditionalGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, latent_dim=100, num_classes=10, d=128, out_channels=3):
        super().__init__()
        
        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Linear layer to process noise + class embedding
        self.input_layer = nn.Linear(latent_dim + num_classes, d*8*4*4)
        self.reshape_size = (-1, d*8, 4, 4)
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),  # [batch, 512, 8, 8]
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),  # [batch, 256, 16, 16]
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*2, out_channels, 4, 2, 1),  # [batch, 3, 32, 32]
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, noise, labels):
        # Process label embedding
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        x = torch.cat([noise, label_embedding], dim=1)
        
        # Process through input layer and reshape
        x = self.input_layer(x)
        x = x.view(self.reshape_size)
        
        # Process through main layers
        x = self.main(x)
        return x

# Better Conditional Discriminator
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, d=64, in_channels=3):
        super().__init__()
        
        # Initial convolution layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, 2, 1),  # [batch, 64, 16, 16]
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Embedding and projection for class conditioning
        self.embed = nn.Embedding(num_classes, d)
        
        # Main convolution layers
        self.main = nn.Sequential(
            nn.Conv2d(d + d, d*2, 4, 2, 1),  # [batch, 128, 8, 8]
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*2, d*4, 4, 2, 1),  # [batch, 256, 4, 4]
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(d*4, 1, 4, 1, 0)  # [batch, 1, 1, 1]
        )
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        h = self.initial(x)
        
        # Get class embeddings and reshape to spatial feature maps
        y_embedding = self.embed(labels)
        y_embedding = y_embedding.view(batch_size, -1, 1, 1)
        y_embedding = y_embedding.expand(batch_size, -1, h.size(2), h.size(3))
        
        # Concatenate features and class embeddings
        h = torch.cat([h, y_embedding], dim=1)
        
        # Process through main layers
        h = self.main(h)
        
        # Return logits, not sigmoid (for better numerical stability)
        return h.view(batch_size, 1)
    

if __name__ == "__main__":
    import torch
    from torchinfo import summary
    
    # Parameters
    batch_size = 32
    latent_dim = 100
    num_classes = 10
    
    # Testing Conditional Generator
    gan = ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes)
    
    # Note: Conditional generator takes noise vector and labels as separate inputs
    noise = torch.randn(batch_size, latent_dim)  # Flattened noise, not 4D
    labels = torch.randint(0, num_classes, (batch_size,))  # Random class labels
    
    # Generate images with the conditional generator
    output = gan(noise, labels)
    print("Generator output shape:", output.shape)  # Should be [32, 3, 32, 32]

    # Testing Conditional Discriminator
    disc = ConditionalDiscriminator(num_classes=num_classes)
    image = torch.randn(batch_size, 3, 32, 32)  # Random fake images
    
    # Discriminate images with conditional discriminator
    output = disc(image, labels)
    print("Discriminator output shape:", output.shape)  # Should be [32, 1, 1, 1]

    # Print model summaries
    print("\nGenerator Summary:")
    summary(gan, input_data=[noise, labels])
    
    print("\nDiscriminator Summary:")
    summary(disc, input_data=[image, labels])