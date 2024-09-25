import os
import re
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def get_last_checkpoint(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter out the checkpoint files and extract their epoch numbers
    checkpoint_files = []
    for file in files:
        match = re.match(r'checkpoint_(\d+)\.pth', file)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((epoch, file))
    if not checkpoint_files:
        return None  # No checkpoint files found
    # Sort by epoch number and return the last one
    checkpoint_files.sort(key=lambda x: x[0])
    last_epoch_file = os.path.join(directory, checkpoint_files[-1][1])
    # get the epoch number from the last checkpoint file
    last_epoch = checkpoint_files[-1][0]
    return last_epoch_file, last_epoch



def show_generated_images(images, n_images=8):
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot(4, 4, i + 1)
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow((img + 1) / 2)  # Rescale from [-1, 1] to [0, 1]
        plt.axis('off')
    plt.show()
    

def save_images(images, output_path, n_images):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(n_images):
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, f'image_{i+1}.png'))
    print(f"Images saved to {output_path}")
    
    
class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Store image paths and corresponding labels
        self.images = []
        self.labels = []
        self.class_to_idx = {}  # Dictionary to map folder names to numerical labels
        
        # Traverse the root directory and subdirectories
        for subdir, _, files in os.walk(root_dir):
            class_name = os.path.basename(subdir)
            if class_name not in self.class_to_idx and len(files) > 0:  # Assign new label if not already assigned
                self.class_to_idx[class_name] = len(self.class_to_idx)  # Assign a new class index
            
            # Collect image paths and labels
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Only consider valid image files
                    self.images.append(os.path.join(subdir, file))
                    self.labels.append(self.class_to_idx[class_name])  # Add corresponding label for the image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label  # Return both the image and label
    
    
def load_dataset(root_dir, image_size) -> any:
    images = []
    labels = []
    
    class_mapping = {
        '0.0.Normal': 0,
        '0.1.Tessellated fundus': 1,
        '0.2.Large optic cup': 2,
        '0.3.DR1': 3,
        '1.0.DR2': 4,
        '1.1.DR3': 5,
        '10.0.Possible glauco': 6,
        '10.1.Optic atrophy': 7,
        '18.Vitreous particles': 8
    }

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).resize((image_size, image_size))
                img = np.array(img) / 255.0  # Normalize to [0, 1]

                label = class_mapping.get(os.path.basename(subdir))
                if label is not None:
                    images.append(img)
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    tensor_images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(tensor_images, tensor_labels)
    return dataset
    
    
def test_classifier(classifier, generator, dataloader, latent_dim) -> tuple:
    classifier.eval()  # Set classifier to evaluation mode
    generator.eval()  # Set generator to evaluation mode

    correct_real = 0
    total_real = 0
    correct_fake = 0
    total_fake = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            # Test on real images
            outputs = classifier(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_real += labels.size(0)
            correct_real += (predicted == labels).sum().item()

            # Test on generated (fake) images
            z = torch.randn(labels.size(0), latent_dim)
            fake_images = generator(z)
            fake_labels = torch.randint(0, 10, (labels.size(0),))  # Fake random labels
            outputs_fake = classifier(fake_images)
            _, predicted_fake = torch.max(outputs_fake.data, 1)
            total_fake += fake_labels.size(0)
            correct_fake += (predicted_fake == fake_labels).sum().item()

    real_accuracy = 100 * correct_real / total_real
    fake_accuracy = 100 * correct_fake / total_fake
    # print(f"Accuracy on real images: {real_accuracy:.2f}%")
    # print(f"Accuracy on generated (fake) images: {fake_accuracy:.2f}%")
    
    return real_accuracy, fake_accuracy


def plot_losses(g_losses, d_losses, c_losses, output_path=None,):
    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(c_losses, label="Classifier Loss")
    plt.legend()
    plt.title("Losses")
    # save image to the output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if output_path:
        plt.savefig(os.path.join(output_path, "losses.png"))
    # plt.show()

def plot_metrics(inception_scores, fid_values, output_path=None):
    # Plot the Inception Scores and FID values
    plt.figure(figsize=(10, 5))
    if inception_scores:
        plt.plot(inception_scores, label="Inception Score")
    if fid_values:
        plt.plot(fid_values, label="FID")
    plt.legend()
    plt.title("Inception Score and FID")
    # save the image to the output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if output_path:
        plt.savefig(os.path.join(output_path, "metrics.png"))
    # plt.show()

