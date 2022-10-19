from torch import nn


class BnActivation(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()

        self.activation = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=num_features, momentum=0.99, affine=False),
        )

    def forward(self, x):
        return self.activation(x)


class LnActivation(nn.Module):
    def __init__(self, normalized_shape=1):
        super().__init__()

        self.activation = nn.Sequential(
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=normalized_shape),
        )

    def forward(self, x):
        return self.activation(x)
