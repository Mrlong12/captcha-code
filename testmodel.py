# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import string
from tqdm import tqdm
from dataset import *
characters = string.digits + string.ascii_letters
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for f in self.features:
            x = f(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    #/home/nls1/.cache/torch/checkpoints/vgg16_bn-6c64b313.pth
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
        model.load_state_dict(torch.load('./vgg16_bn.pth'))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:7]),
                 nn.Sequential(*list_feature[7:14]),
                 nn.Sequential(*list_feature[14:24]),
                 nn.Sequential(*list_feature[24:34]),
                 nn.Sequential(*list_feature[34:43])]
    model.features = nn.ModuleList(_features)
    return model
def decode(sequence):
    s = ''.join([characters[x] for x in sequence])
    return s
class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.cnn = nn.Sequential(*vgg16_bn(False).features, nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.fc = nn.Sequential(
            nn.Linear(in_features=3584, out_features=1024),
            nn.Dropout(0.25),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=4 * n_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = x.reshape(x.shape[0], -1, 4)
        x = F.softmax(x, dim=1)
        return x
def test(model,dataloader):
    model.eval()
    result = []
    with tqdm(dataloader) as pbar, torch.no_grad():
        for batch_index, (data,name) in enumerate(pbar):
            data = data
            output = model(data)
            output_argmax = output.detach().argmax(dim=1)
            output_argmax = output_argmax.cpu().numpy()
            for pred in output_argmax:
                B = decode(pred)
                print('{}--{}'.format(name[0],B))
                result.append(B)
    return result
def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件

    width, height, n_len, n_classes = 120, 40, 4, len(characters)
    model = Model(n_classes, input_shape=(3, height, width))
    model_loaded = torch.load('./final_ctc.pth', map_location='cpu')
    model.load_state_dict(model_loaded)
    pass
    test_loader = get_loader(path=testpath, batch_size=1, mode='test')
    print("reading end!")
    labels = test(model,test_loader)
    # the format of result-file
    # 这里可以生成结果文件
    ids = [str(x) + ".jpg" for x in range(1, 5001)]
    df = pd.DataFrame([ids, labels]).T
    df.columns = ['ID', 'label']
    return df
