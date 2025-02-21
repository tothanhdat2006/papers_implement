import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()

    def forward(self, X):
        shortcut = X.clone()

        X = self.conv1(X)
        X = self.batch_norm1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.batch_norm2(X)
        X += self.downsample(shortcut)
        X = self.relu(X)
        
        return X


class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_list, n_classes):
        super(ResNet, self).__init__()
        assert len(n_blocks_list) == 4, f'ResNet __init__ error: n_blocks_list expected 4 elements, found {len(n_blocks_list)} instead\n'

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self.create_layer(residual_block, 64, 64, n_blocks_list[0], 1)
        self.conv3_x = self.create_layer(residual_block, 64, 128, n_blocks_list[1], 2)
        self.conv4_x = self.create_layer(residual_block, 128, 256, n_blocks_list[2], 2)
        self.conv5_x = self.create_layer(residual_block, 256, 512, n_blocks_list[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)
    
    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        blocks.append(residual_block(in_channels, out_channels, stride))
        for idx in range(1, n_blocks):
            blocks.append(residual_block(out_channels, out_channels, 1))
        
        return nn.Sequential(*blocks)

    def forward(self, X):
        X = self.conv1(X)
        X = self.batch_norm1(X)
        X = self.maxpool(X)
        X = self.relu(X)

        X = self.conv2_x(X)
        X = self.conv3_x(X)
        X = self.conv4_x(X)
        X = self.conv5_x(X)

        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc1(X)

        return X
