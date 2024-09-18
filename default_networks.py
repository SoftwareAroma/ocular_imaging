"""
    Some of the default network for the Tripple GAN configuration is in this file.
    new configurations can be added and used as *args in the initiation of the networks
    in the main.py file.
"""

import torch.nn as nn

def get_default_disc_layers(input_dim:int=128*128*3) -> list[any]:
    """
        Get the default discriminator layers
        Args:
            input_dim (int): The input dimension of the input layer of the network

        Returns:
            list: Layers
    """
    return [
        nn.Linear(input_dim, 1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ]
    

def  get_default_gen_layers(input_dim:int=100, output_dim:int=128*128*3) -> list[any]:
    """
        Get the default generator layers (this can be modified to suit the need of the program)
        Args:
            input_dim (int): The input dimension for the input layer
            output_dim (int): The output dimension of the output layer
        Returns:
            list[any]: The list of layers created for the network
    """
    return [
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, output_dim),
        nn.Tanh()
    ]
    
    
def get_default_classifier_layers(input_dim:int = 128*128*3, num_classes:int = 10) -> list[any]:
    """
        Get the default classifier layers for the network
        Args:
            input_dim (int): the input dimension of the input layer
            num_classes (int): the number of classes
        Returns:
            list[any]: the list of layers for the classifier
    """
    return [
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    ]