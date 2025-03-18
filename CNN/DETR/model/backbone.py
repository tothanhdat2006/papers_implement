import torch
import torch.nn as nn

cfg_blocks = {
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
}

cfg_channels = {
    'ResNet18': [64, 64, 128, 256],
    'ResNet34': [64, 64, 128, 256],
    'ResNet50': [64, 256, 512, 1024],
    'ResNet101': [64, 256, 512, 2048],
    'ResNet152': [64, 256, 512, 2048],
}

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        shortcut = X.clone()

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X += self.downsample(shortcut)
        X = self.relu(X)
        return X
        

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, X):
        shortcut = X.clone()

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X += self.downsample(shortcut)
        X = self.relu(X)
        return X

class ResNet(nn.Module):
    def __init__(self, in_channels, residual_block, out_channels_list, n_blocks_list, n_classes, features_only=False):
        super(ResNet, self).__init__()
        assert len(n_blocks_list) == 4, \
            f'ResNet __init__ error: n_blocks_list expected 4 elements, found {len(n_blocks_list)} instead\n'
        assert len(out_channels_list) == 4, \
            f'ResNet __init__ error: out_channels_list expected 4 elements, found {len(out_channels_list)} instead\n'
        
        self.features_only = features_only
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self.create_layer(residual_block, 64, out_channels_list[0], n_blocks_list[0], 1)
        self.conv3_x = self.create_layer(residual_block, out_channels_list[0] * residual_block.expansion, out_channels_list[1], n_blocks_list[1], 2)
        self.conv4_x = self.create_layer(residual_block, out_channels_list[1] * residual_block.expansion, out_channels_list[2], n_blocks_list[2], 2)
        self.conv5_x = self.create_layer(residual_block, out_channels_list[2] * residual_block.expansion, out_channels_list[3], n_blocks_list[3], 2)

        if features_only:
            self.avgpool = None
            self.flatten = None
            self.fc1 = None
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(512 * residual_block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)
        
    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(residual_block(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(residual_block(out_channels * residual_block.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)

        X = self.conv2_x(X)
        X = self.conv3_x(X)
        X = self.conv4_x(X)
        X = self.conv5_x(X)

        if self.features_only:
            return X
        
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc1(X)

        return X

def build_backbone(model_name: str, n_classes: int) -> ResNet:
    assert model_name in cfg_blocks, f'build_backbone error: model_name expected in {cfg_blocks.keys()}, found {model_name} instead\n'
    if model_name == 'ResNet18':
        return ResNet(in_channels=3, residual_block=BasicBlock, 
                      out_channels_list=cfg_channels[model_name], n_blocks_list=cfg_blocks[model_name], n_classes=n_classes, features_only=True)
    elif model_name == 'ResNet34':
        return ResNet(in_channels=3, residual_block=BasicBlock, 
                      out_channels_list=cfg_channels[model_name], n_blocks_list=cfg_blocks[model_name], n_classes=n_classes, features_only=True)
    elif model_name == 'ResNet50':
        return ResNet(in_channels=3, residual_block=Bottleneck, 
                      out_channels_list=cfg_channels[model_name], n_blocks_list=cfg_blocks[model_name], n_classes=n_classes, features_only=True)
    elif model_name == 'ResNet101':
        return ResNet(in_channels=3, residual_block=Bottleneck, 
                      out_channels_list=cfg_channels[model_name], n_blocks_list=cfg_blocks[model_name], n_classes=n_classes, features_only=True)
    elif model_name == 'ResNet152':
        return ResNet(in_channels=3, residual_block=Bottleneck, 
                      out_channels_list=cfg_channels[model_name], n_blocks_list=cfg_blocks[model_name], n_classes=n_classes, features_only=True)
