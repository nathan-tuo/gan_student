# Write your code for the following task underneath the comment lines! 
#
#
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from gan_model import ConditionalGenerator, ConditionalDiscriminator
from tqdm import tqdm 
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import os

def train_cgan(generator, discriminator, train_loader, num_epochs, latent_dim, device):
    
    generator = generator.to(...)
    discriminator = discriminator.to(...)
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    fixed_noise = torch.randn(100, latent_dim, device=device)
    fixed_labels = torch.tensor([i for i in range(10) for _ in range(10)], device=device)
    
    for epoch in range(num_epochs):
        ....train()
        ....train()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        batches_done = 0
        
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Move data to device


            # -----------------------------------------
            #  Create Train Discriminator (with label noise)
            # -----------------------------------------

            # Use zero_grad() for optimizer
            
            real_labels_noise = real_labels - 0.1 * torch.rand(real_labels.size(), device=device)
            fake_labels_noise = fake_labels + 0.1 * torch.rand(fake_labels.size(), device=device)
            
            # Process real images (no detach needed for real images)

            
            # Generate fake images

            
            # Process fake images
            
            
            # Calculate the total discriminator loss


            # Only update discriminator if it's not too strong


            # If discriminator is too accurate, skip update. Allow discriminator to update only if it's not too good

            
            # -----------------------------------------
            #  Create Train Generator (every other batch)
            # -----------------------------------------

            if i % 2 == 0:  
                
                # Use zero_grad() for optimizer
                

                # Generate new batch of fake images

                
                # Get discriminator predictions


                # Generator wants discriminator to think its images are real


                # Update generator

                # You are done! Rest of the code in 4.trainer.py is complete!

            
            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()
            batches_done += 1
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                      f"D_loss: {loss_d.item():.4f}, G_loss: {loss_g.item():.4f}, "
                      f"D_acc: {d_accuracy:.2f}")
        
        # Update learning rates
        ....step()
        ....step()
        
        # Calculate average epoch losses
        avg_d_loss = total_d_loss / batches_done
        avg_g_loss = total_g_loss / batches_done
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
        
        # Save sample images periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Generate sample images
            generator.eval()
            with torch.no_grad():
                sample_images = generator(fixed_noise, fixed_labels)
                save_path = f"samples/epoch_{epoch+1}.png"
                
                import torchvision.utils as vutils
                vutils.save_image(sample_images.data[:100], save_path, 
                                 normalize=True, nrow=10)
                print(f"Saved samples to {save_path}")
        
        # Save models periodically
        if (epoch+1) % 10 == 0 or (epoch+1) == num_epochs:
            torch.save(generator.state_dict(), f'checkpoints/cgan_generator_epoch{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/cgan_discriminator_epoch{epoch+1}.pth')
    
    print("Training completed!")

# Usage example - this would go in your trainer.py file
if __name__ == "__main__":

    # Parameters
    batch_size = 64
    latent_dim = 100
    num_classes = 10
    num_epochs = ...  # More epochs for better training

    # Set device
    device = torch.device("..." if torch.cuda.is_available() else "...")
    print(f"Using device: {device}")

    # Initialize models
    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes)
    discriminator = ConditionalDiscriminator(num_classes=num_classes)

    # Use Binary Cross Entropy with Logits for better stability
    # (combines sigmoid and BCE in a single operation)
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Better optimizers with different learning rates
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Learning rate schedulers to stabilize training
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    # Train the model
    train_cgan(generator, discriminator, train_loader, num_epochs, latent_dim, device)
