import os
import torch
import warnings
import tempfile
import shutil
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.model import TripleGAN
from src.networks import (
    Generator, 
    Discriminator, 
    Classifier
)
from src.utils import (
    FundusDataset, 
    FundusDatasetOne, 
    load_dataset, 
    plot_losses, 
    plot_metrics, 
    save_images, 
    get_last_checkpoint
)
from default_networks import (
    get_default_classifier_layers,
    get_default_disc_layers,
    get_default_gen_layers
)
from options import parse_args
from torchvision.models import inception_v3
from pytorch_fid import fid_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image



warnings.filterwarnings("ignore")

os.environ['TORCH_HOME'] = './pre-trained/' #setting the environment variable

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()


# Function to calculate Inception Score
def calculate_inception_score(gen_imgs, splits=10):
    with torch.no_grad():
        pred = inception_model(gen_imgs)
        pred = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()
        split_scores = []

        for k in range(splits):
            part = pred[k * (pred.shape[0] // splits): (k + 1) * (pred.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            split_scores.append(np.exp(kl))

        return np.mean(split_scores), np.std(split_scores)


# Function to calculate FID
def calculate_fid(real_imgs, gen_imgs, batch_size=1):
    # Create temporary directories
    real_dir = tempfile.mkdtemp()
    gen_dir = tempfile.mkdtemp()

    # Save real images
    for i, img in enumerate(real_imgs):
        save_image(img, os.path.join(real_dir, f"real_{i}.png"))

    # Save generated images
    for i, img in enumerate(gen_imgs):
        save_image(img, os.path.join(gen_dir, f"gen_{i}.png"))

    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir], 
        batch_size=batch_size, 
        device=device, dims=2048
    )

    # Clean up temporary directories
    shutil.rmtree(real_dir)
    shutil.rmtree(gen_dir)

    return fid_value

# def calculate_fid(real_imgs, gen_imgs, batch_size=64):
#     fid_value = fid_score.calculate_fid_given_paths(
#         [real_imgs, gen_imgs], 
#         batch_size=batch_size, 
#         device=device, dims=2048
#     )
#     return fid_value

# Function to train or test the model
def train_or_test(options):
    
    if options.train:
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((options.image_size, options.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print("Loading dataset...")
        dataset = FundusDatasetOne(options.root_dir, transform=transform)
        print(f"Dataset loaded | Number of images loaded: {[len(dataset)]}")
        
        dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)
    else:
        print("Preparing model for testing")

    gen_layers = get_default_gen_layers(
        input_dim=options.latent_dim, 
        output_dim=128*128*3
    )
    disc_layers = get_default_disc_layers(
        input_dim=128*128*3
    )
    classifier_layers = get_default_classifier_layers(
        input_dim=128*128*3, 
        num_classes=options.num_classes
    )
    
    print("Creating netowrk")
    generator = Generator(
        input_dim=options.latent_dim, 
        layers=gen_layers
    )
    discriminator = Discriminator(
        layers=disc_layers
    )
    classifier = Classifier(
        num_classes=options.num_classes, 
        layers=classifier_layers
    )

    optimizer_G = torch.optim.Adam(
        generator.parameters(), 
        lr=options.lr_G, 
        betas=options.beta_G
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=options.lr_D, 
        betas=options.beta_D
    )
    optimizer_C = torch.optim.Adam(
        classifier.parameters(), 
        lr=options.lr_C, 
        betas=options.beta_C
    )

    gan_model = TripleGAN(
        generator=generator,
        discriminator=discriminator,
        classifier=classifier,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        optimizer_C=optimizer_C,
        options=options,
        latent_dim=options.latent_dim
    )
    
    
    g_losses = []
    d_losses = []
    c_losses = []
    inception_scores = []
    fid_values = []

    if options.train:
        print("Starting training...")
        for epoch in range(options.num_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                d_loss, g_loss, c_loss, fake_images = gan_model.train_step(imgs, labels)
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                c_losses.append(c_loss.item())
                
            #  TODO: calculate metrics
            # inception_score, _ = calculate_inception_score(fake_images)
            # fid_value = calculate_fid(imgs, fake_images, options.batch_size)
            # inception_scores.append(inception_score)
            # fid_values.append(fid_value)

            print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
            # print(f"Inception Score: {inception_score} FID: {fid_value}") # Inception Score: {inception_score}
            # create the checkpoint directory if it does not exist
            if not os.path.exists(options.checkpoint_dir):
                os.makedirs(options.checkpoint_dir)
            # TODO: uncomment the following lines before training if you want to save the model checkpoints
            if epoch % 500 == 0:
                checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
                gan_model.save_checkpoint(checkpoint_path)
        #TODO: uncomment the following lines before training if you want to save the model
        # #Save the model after training
        if not os.path.exists(options.output_path):
            os.makedirs(options.output_path)
        gan_model.save_model(os.path.join(options.output_path, options.model_path))
        
    elif options.continue_training:
        print("Continuing training from the last checkpoint...")
        # get the last checkpoint in the checkpoint directory
        last_checkpoint, last_epoch = get_last_checkpoint(options.checkpoint_dir)
        if os.path.exists(last_checkpoint):
            print(f"Loading checkpoint from {last_checkpoint}")
            gan_model.load_model(last_checkpoint)
            # Continue training from the last epoch
            for epoch in range(last_epoch, options.num_epochs):
                for i, (imgs, labels) in enumerate(dataloader):
                    d_loss, g_loss, c_loss, fake_images = gan_model.train_step(imgs, labels)
                    g_losses.append(g_loss.item())
                    d_losses.append(d_loss.item())
                    c_losses.append(c_loss.item())
                    
                # TODO: Calculate metrics
                # inception_score, _ = calculate_inception_score(fake_images)
                # fid_value = calculate_fid(imgs, fake_images, options.batch_size)
                # inception_scores.append(inception_score)
                # fid_values.append(fid_value)

                print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
                # print(f"Inception Score: {inception_score}, FID: {fid_value}")
                # TODO: uncomment the following lines before training if you want to save the model checkpoints
                if epoch % 500 == 0:
                    checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
                    gan_model.save_checkpoint(checkpoint_path)
    else:
        print("loading model...")
        gan_model.load_model(os.path.join(options.output_path, options.model_path))
        print("generating images ")
        images = gan_model.generate_images(options.n_images)
        save_images(images, options.output_path, options.n_images)
        # print(f"images saved to {options.output_dir}")
        
    # export the losses and the metrics to a csv file
    if options.train or options.continue_training:
        df = pd.DataFrame({
            'g_losses': g_losses,
            'd_losses': d_losses,
            'c_losses': c_losses,
            'inception_scores': inception_scores,
            'fid_values': fid_values
        })
        df.to_csv(os.path.join(options.output_path, 'metrics.csv'), index=False)
        
        plot_losses(g_losses, d_losses, c_losses, options.output_path)
        plot_metrics(inception_scores, fid_values, options.output_path)
        

if __name__ == '__main__':
    args = parse_args()
    train_or_test(args)
    