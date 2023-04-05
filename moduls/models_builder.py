from torch import nn

def hello():
    print("Hello this is models_builder.py file.")

class Fashion_modelv1(nn.Module):
    """
    NonLinear model without any Convolutional layers
    Input shape: is number after Flatten layer
    Example: IMG size 64x64 -> input shape 64*64*Color
    """
    def __init__(self,
                 in_features:int,
                 hidden_units:int,
                 out_features:int):
        super().__init__()
        self.stacked_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=out_features),
            nn.ReLU()
        )
    def forward(self,x):
        return self.stacked_layers(x)

class Fashion_CNNv2(nn.Module):
    """
    Classic CNN from TinyVGG with little to none modifications.
    Hard coded Flatten->Linear layer shape
    Expected size: 64x64
    """
    def __init__(self,
                 in_features:int,
                 hidden_units:int,
                 out_features:int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=out_features)
        )
    def forward(self,x):
        y = self.conv_block1(x)
        #print(y.shape)
        y = self.conv_block2(y)
        #print(y.shape)
        #print(self.conv_block2.parameters())
        y = self.classifier(y)
        return y

class baseline_model(nn.Module):
    """
    Linear model without Convolutional layers
    Input shape: is number after Flatten layer
    Example: IMG size 64x64 -> input shape 64*64*Color
    """
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)

class Tiny_VGG(nn.Module):
    """
    Replicated Tiny VGG model witout modifications
    Hard coded Flatten->Linear layer shape
    Expected size: 64x64
    """
    def __init__(self,
                 in_features,
                 hidden_units,
                 out_features):
        super().__init__()

        self.block_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units , kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units , kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.block_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units , kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units , kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,out_features=out_features)

        )

    def forward(self,x):
        return self.block_classifier(self.block_conv2(self.block_conv1(x)))


