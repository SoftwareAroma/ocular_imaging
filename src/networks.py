import torch.nn as nn

class Generator(nn.Module):
    def __init__(
        self,
        layers=None,
        input_dim=100,
        output_dim=128*128*3
    ):
        """The generator network

        Args:
            layers (list, optional): The list of layers to create the network. Defaults to None.
            input_dim (int, optional): The input dimension. Defaults to 100.
            output_dim (int, optional): The output dimention of the images generated. Defaults to 128*128*3.
        """
        super(Generator, self).__init__()
        if layers is None:
            layers = [
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim),
                nn.Tanh()
            ]
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 128, 128)  # Reshape to image dimensions
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        layers=None,
        input_dim=128*128*3
    ):
        """
            The discriminator network

            Args:
                layers (list, optional): The list of layers to create the discriminator network. Defaults to None.
                input_dim (int, optional): The input dimention of the images. Defaults to 128*128*3.
        """
        super(Discriminator, self).__init__()
        if layers is None:
            layers = [
                nn.Linear(input_dim, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        layers=None,
        input_dim=128*128*3,
        num_classes=10
    ):
        """
            The classifier network

            Args:
                layers (list[nn], optional): The list of layers to use to create the classifier network. Defaults to None.
                input_dim (int, optional): The input dimention of the images. Defaults to 128*128*3.
                num_classes (int, optional): The number of classes in the dataset. Defaults to 10.
        """
        super(Classifier, self).__init__()
        if layers is None:
            layers = [
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x
    