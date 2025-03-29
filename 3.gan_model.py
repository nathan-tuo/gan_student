# Write your code for the following task underneath the comment lines! 
#
#
#
import torch
import torch.nn as nn
from torchinfo import summary
from huggingface_hub import PyTorchModelHubMixin

class ConditionalGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, latent_dim=100, num_classes=10, d=128, out_channels=3):
        super().__init__()
        
        # Embedding layer for class labels
        
        # Linear layer to process noise + class embedding

        # reshape to (32, 1024, 4, 4)
        
        # Transposed convolution layers using Sequential
        
        
    def forward(self, noise, labels):
        
        # Process label embedding
        
        x = torch.cat([...], dim=1)
        
        x = self.input_layer(...)
        x = x.view(...)

        x = self.main(x)
        return x

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, d=64, in_channels=3):
        super().__init__()
        
        self.initial = nn.Sequential(

            # Conv2d layer and a LeakyReLU layer

        )
        
        # Embedding layer


        # Main convolution layers using Sequential
        self.main = nn.Sequential(

        )

        # You are done! Rest of the code in 3.gan_model.py is complete!
        
    def forward(self, x, labels):
        batch_size = x.size(0) # 32, but can be different
        first_conv_out = self.initial(x) # Output shape (32, 64, 16, 16)
        
        # Get class embeddings and reshape to spatial feature maps
        y_embedding = self.embed(labels) # Outputs (32, 64)

        y_embedding = y_embedding.view(batch_size, -1, 1, 1) # Outputs (32, 64, 1, 1)
        y_embedding = y_embedding.expand(batch_size, -1, first_conv_out.size(2), first_conv_out.size(3)) # Shape (32, 64, 16, 16)
        
        # Concatenate features and class embeddings
        first_conv_out = torch.cat([first_conv_out, y_embedding], dim=1)
        
        # Process through main layers
        output = self.main(first_conv_out) # Outputs (32, 1, 1, 1)
        
        # Return logits, not sigmoid (for better numerical stability)
        return output.view(batch_size, 1)


if __name__ == "__main__":        
    # Testing Conditional Generator
    gan = ConditionalGenerator(latent_dim=100, num_classes=10)
    
    # Note: Conditional generator takes noise vector and labels as separate inputs
    noise = torch.randn(32, 100)  # Flattened noise, not 4D
    labels = torch.randint(0, 10, (32,))  # Random class labels
    
    # Generate images with the conditional generator
    output = gan(noise, labels)
    print("Generator output shape:", output.shape)  # Should be [32, 3, 32, 32]

    # Testing Conditional Discriminator
    disc = ConditionalDiscriminator(num_classes=10)
    image = torch.randn(32, 3, 32, 32)  # Random fake images
    
    # Discriminate images with conditional discriminator
    output = disc(image, labels)
    print("Discriminator output shape:", output.shape)  # Should be [32, 1, 1, 1]

    # summary(gan, input_data=[noise, labels])
    
    # summary(disc, input_data=[image, labels])