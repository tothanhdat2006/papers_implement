import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

'''
600,600,3   -> 600,600,64 -> 600,600,64 -> 300,300,64 
            -> 300,300,128 -> 300,300,128 -> 150,150,128 
            -> 150,150,256 -> 150,150,256 -> 150,150,256 -> 75,75,256 
            -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 
            ->  37,37,512 -> 37,37,512 -> 37,37,512
'''
cfg = [64, 64, 'M', 
       128, 128, 'M', 
       256, 256, 256, 'M', 
       512, 512, 512, 'M', 
       512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, features, n_classes=1000, init_weight=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

        if(init_weight):
            self._init_weight()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_features(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1)
            layers += [conv2d]
            if batch_norm:
                layers += [nn.BatchNorm2d(v)]
            layers += [nn.ReLU(True)]
            
            in_channels = v
    
    return nn.Sequential(*layers)

def custom_vgg16(pretrained=False):
    model = VGG(make_features(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    features = list(model.features)[:30]
    classifier = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    
    return features, classifier