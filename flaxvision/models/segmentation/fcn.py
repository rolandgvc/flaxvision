from flax import linen as nn
import numpy as np

class FCNHead(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, inputs, train=False):
        inter_channels = np.shape(inputs)[-1] // 4
        x = nn.Conv(inter_channels, (3,3), padding='SAME', use_bias=False, name="conv1")(inputs)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.Conv(self.channels, (1,1), padding='VALID', use_bias=True, name="conv2")(x)

        return x


'''
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)
'''
