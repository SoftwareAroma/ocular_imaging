import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='GAN with Classifier for occular image re-generation'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='Continue training from the last checkpoint'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='triple_gan_model.pth',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--n_images',
        type=int,
        default=8,
        help='Number of images to generate during testing'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='output',
        help='The directory to save the generated images'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Path to save the model checkpoints'
    )
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=128, 
        help='Size of the input images'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=100, 
        help='Number of epochs for training'
        )
    parser.add_argument(
        '--lr_G', 
        type=float, 
        default=0.0002, 
        help='Learning rate for generator'
    )
    parser.add_argument(
        '--lr_D', 
        type=float, 
        default=0.0002, 
        help='Learning rate for discriminator'
    )
    parser.add_argument(
        '--lr_C', 
        type=float, 
        default=0.0002, 
        help='Learning rate for classifier'
    )
    parser.add_argument(
        '--latent_dim', 
        type=int, 
        default=100, 
        help='Dimension of the latent vector'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1000,
        help='Number of classes in the dataset'
    )
    parser.add_argument(
        '--beta_G', 
        type=float,
        default=(0.5, 0.999),
        help='Beta for generator optimizer'
    )
    parser.add_argument(
        '--beta_D',
        type=float,
        default=(0.5, 0.999),
        help='Beta for discriminator optimizer'
    )
    parser.add_argument(
        '--beta_C',
        type=float,
        default=(0.5, 0.999),
        help='Beta for classifier optimizer'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='-1',
        help='gpu ids: e.g. 0 -> for one GPU  0,1,2, -> for multiple GPUs, -1 -> for CPU'
    )
    args = parser.parse_args()
    return args
