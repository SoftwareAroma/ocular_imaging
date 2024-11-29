import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch

os.environ['TORCH_HOME'] = './pre-trained/'  # Set the environment variable

class DDPMFundusTrainer:
    def __init__(self, data_dir, batch_size=8, lr=1e-4, epochs=10, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # Initialize dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Load pre-trained DDPM model and scheduler
        self.model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
        
        # Initialize optimizer and FID metric
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.fid = FrechetInceptionDistance().to(self.device)
        
        
class LimitedImageFolderDataset(Dataset):
    def __init__(self, root, transform=None, max_images=40000):
        self.transform = transform
        self.max_images = max_images

        # Gather all image file paths from subdirectories
        self.image_paths = []
        for subdir, _, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                    self.image_paths.append(os.path.join(subdir, file))

        # Limit to the maximum number of images
        if len(self.image_paths) > self.max_images:
            self.image_paths = random.sample(self.image_paths, self.max_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = datasets.folder.default_loader(image_path)  # Load image
        if self.transform:
            image = self.transform(image)
        return image


class DDPMFundusTrainerOne:
    def __init__(self, data_dir, batch_size=8, lr=1e-4, epochs=10, device=None, max_images=40000):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # Initialize dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        dataset = LimitedImageFolderDataset(root=data_dir, transform=transform, max_images=max_images)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Load pre-trained DDPM model and scheduler
        self.model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        # Initialize optimizer and FID metric
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.fid = FrechetInceptionDistance().to(self.device)
        


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device)

                # Sample random timesteps for each image in the batch
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],), device=self.device).long()

                # Add noise to images
                noise = torch.randn_like(images)
                noisy_images = self.scheduler.add_noise(images, noise, timesteps)

                # Predict noise
                noise_pred = self.model(noisy_images, timesteps).sample

                # Compute loss and backpropagation
                loss = F.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item()}")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        self.model.save_pretrained(f"./ddpm-finetuned-fundus-epoch{epoch}")
        self.scheduler.save_pretrained(f"./ddpm-scheduler-fundus-epoch{epoch}")
        print(f"Checkpoint saved for epoch {epoch}")

    def load_checkpoint(self, model_path, scheduler_path):
        self.model = UNet2DModel.from_pretrained(model_path).to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        print("Checkpoint loaded.")

    def save_model(self, model_path="./ddpm-finetuned-fundus-final"):
        self.model.save_pretrained(model_path)
        print("Model saved.")

    def load_model(self, model_path="./ddpm-finetuned-fundus-final"):
        self.model = UNet2DModel.from_pretrained(model_path).to(self.device)
        print("Model loaded for inference.")

    def inference(self):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.scheduler).to(self.device)
        for i, (real_batch, _) in enumerate(self.dataloader):
            real_batch = real_batch.to(self.device)
            for j in range(5):  # Generate 5 images per batch
                generated_image = pipeline().images[0].convert("RGB")
                generated_image_tensor = transforms.ToTensor()(generated_image).unsqueeze(0).to(self.device)
                
                # Save generated image
                generated_image.save(f"./synthetic_fundus_image_batch{i}_image{j}.png")
                print(f"Generated Image {i}_{j} saved!")

                # Evaluation: PSNR and SSIM calculations
                psnr = self.calculate_psnr(real_batch[0], generated_image_tensor[0])
                print(f"PSNR for Image {i}_{j}: {psnr}")

                ssim_value = self.calculate_ssim(real_batch[0], generated_image_tensor[0])
                print(f"SSIM for Image {i}_{j}: {ssim_value}")

        # Compute final FID score after generating all images
        fid_score = self.fid.compute()
        print(f"Final FID Score: {fid_score.item()}")

    @staticmethod
    def calculate_psnr(real_image, generated_image):
        mse = F.mse_loss(real_image, generated_image)
        psnr = 10 * torch.log10(1 / mse)
        return psnr.item()

    @staticmethod
    def calculate_ssim(real_image, generated_image):
        real_image_np = real_image.cpu().numpy().transpose(1, 2, 0)
        generated_image_np = generated_image.cpu().numpy().transpose(1, 2, 0)
        return ssim(real_image_np, generated_image_np, multichannel=True)


if __name__ == "__main__":
    trainer = DDPMFundusTrainerOne(data_dir="/mnt/data/RiskIntel/ocular_imaging/datasets")
    # trainer = DDPMFundusTrainer(data_dir="/mnt/data/RiskIntel/ocular_imaging/datasets/1000images")
    trainer.train()
    trainer.save_model()
    trainer.load_model()
    trainer.inference()

# # Example usage:
# trainer = DDPMFundusTrainer(data_dir="./datasets/fundusimage1000/1000images")
# trainer.train()
# trainer.save_model()
# trainer.load_model()
# trainer.inference()
