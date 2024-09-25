import torch
import torch.nn as nn
from torchsummary import summary

class TripleGAN(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        classifier,
        optimizer_G,
        optimizer_D,
        optimizer_C,
        options,
        criterion_D=nn.BCELoss(),
        criterion_C=nn.CrossEntropyLoss(),
        latent_dim=100,
        device = None
        
    ):
        super(TripleGAN, self).__init__()
        self.options = options
        
        if device is not None:
            self.device = device
            
        self.device = torch.device(
            f"cuda:{self.options.gpu_ids[0]}" if torch.cuda.is_available() and self.options.gpu_ids != '-1' else 'cpu'
        )
        
        # Move networks to the selected device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.classifier = classifier.to(self.device)
        
        # If using multiple GPUs, wrap networks in DataParallel
        if len(self.options.gpu_ids) > 1 and self.device.type == 'cuda':
            self.generator = nn.DataParallel(
                self.generator, device_ids=[
                    int(id) for id in self.options.gpu_ids.split(',')
                ]
            )
            self.discriminator = nn.DataParallel(
                self.discriminator, device_ids=[
                    int(id) for id in self.options.gpu_ids.split(',')
                ]
            )
            self.classifier = nn.DataParallel(
                self.classifier, 
                device_ids=[
                    int(id) for id in self.options.gpu_ids.split(',')
                ]
            )
            
        self.criterion_D = criterion_D
        self.criterion_C = criterion_C
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_C = optimizer_C
        self.latent_dim = latent_dim
        
    def get_networks(self):
        return self.generator, self.discriminator, self.classifier
    
    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'classifier_state_dict': self.classifier.state_dict()
        }, path)
        

    def save_checkpoint(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_C_state_dict': self.optimizer_C.state_dict()
        }, path)
    
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    
    def generate_images(self, n_images):
        z = torch.randn(n_images, self.latent_dim)
        return self.generator(z)

    def forward(self, real_images):
        batch_size = real_images.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1, device=real_images.device)
        fake_labels = torch.zeros(batch_size, 1, device=real_images.device)

        # Discriminator output
        d_output_real = self.discriminator(real_images)

        # Generate fake images
        z = torch.randn(batch_size, 100, device=real_images.device)
        fake_images = self.generator(z)
        d_output_fake = self.discriminator(fake_images)

        # Classifier output
        c_output_real = self.classifier(real_images)
        c_output_fake = self.classifier(fake_images)

        return d_output_real, d_output_fake, fake_images, real_labels, fake_labels, c_output_real, c_output_fake

    def train_step(self, real_images, labels):
        batch_size = real_images.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1, device=real_images.device)
        fake_labels = torch.zeros(batch_size, 1, device=real_images.device)

        # Train discriminator
        outputs = self.discriminator(real_images)
        d_loss_real = self.criterion_D(outputs, real_labels)
        
        z = torch.randn(batch_size, 100, device=real_images.device)
        fake_images = self.generator(z)
        outputs = self.discriminator(fake_images.detach())  # Detach to prevent backprop
        d_loss_fake = self.criterion_D(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        # Train generator
        z = torch.randn(batch_size, 100, device=real_images.device)  # Ensure the same device
        fake_images = self.generator(z)
        outputs = self.discriminator(fake_images)  # No detach here, we want gradients
        g_loss = self.criterion_D(outputs, real_labels)

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Train classifier
        outputs = self.classifier(real_images)
        c_loss_real = self.criterion_C(outputs, labels)

        fake_labels_gen = self.classifier(fake_images.detach())  # Detach to prevent backprop
        c_loss_fake = self.criterion_C(fake_labels_gen, torch.argmax(fake_labels_gen, dim=1))

        c_loss = c_loss_real + c_loss_fake

        self.optimizer_C.zero_grad()
        c_loss.backward()
        self.optimizer_C.step()

        return d_loss, g_loss, c_loss, fake_images
    
