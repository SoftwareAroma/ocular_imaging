import os
import torch
from torch.utils.data import DataLoader
from src.networks import Generator, Discriminator, Classifier
from src.model import TripleGAN
from src.utils import calculate_fid, calculate_inception_score, get_inception_model, load_dataset, save_images, get_last_checkpoint
from default_networks import (
    get_default_classifier_layers,
    get_default_disc_layers,
    get_default_gen_layers
)
from options import parse_args
import warnings
warnings.filterwarnings("ignore")

os.environ['TORCH_HOME'] = './pre-trained/' #setting the environment variable

# Function to train or test the model
def train_or_test(options):
    if options.train:
        print("Loading dataset...")
        dataset = load_dataset(
            options.root_dir,
            options.image_size
        )
        print(f"Dataset loaded...") # {len(dataset)} images
        dataloader = DataLoader(
            dataset, 
            batch_size=options.batch_size, 
            shuffle=True
        )
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
    
    inception_model=get_inception_model(),
    
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
                
            # Calculate metrics
            inception_score, _ = calculate_inception_score(inception_model, fake_images)
            fid_value = calculate_fid(imgs, fake_images)
            inception_scores.append(inception_score)
            fid_values.append(fid_value)

            print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
            print(f"Inception Score: {inception_score} FID: {fid_value}")
            # create the checkpoint directory if it does not exist
            if not os.path.exists(options.checkpoint_dir):
                os.makedirs(options.checkpoint_dir)
            # TODO: uncomment the following lines before training if you want to save the model checkpoints
            # if epoch % 100 == 0:
            #     checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
            #     gan_model.save_checkpoint(checkpoint_path)
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
                    
                # Calculate metrics
                inception_score, _ = calculate_inception_score(inception_model, fake_images)
                fid_value = calculate_fid(imgs, fake_images)
                inception_scores.append(inception_score)
                fid_values.append(fid_value)

                print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
                print(f"Inception Score: {inception_score} FID: {fid_value}")
                # TODO: uncomment the following lines before training if you want to save the model checkpoints
                # if epoch % 100 == 0:
                #     checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
                #     gan_model.save_checkpoint(checkpoint_path)
    else:
        print("loading model...")
        gan_model.load_model(os.path.join(options.output_path, options.model_path))
        print("generating images ")
        images = gan_model.generate_images(options.n_images)
        save_images(images, options.output_path, options.n_images)
        # print(f"images saved to {options.output_dir}")
        

if __name__ == '__main__':
    args = parse_args()
    train_or_test(args)
    