from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet18':
            model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pret else None)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pret else None)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pret else None)
        if option == 'resnet101':
            model_ft = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pret else None)
        if option == 'resnet152':
            model_ft = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pret else None)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x


class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if pret else None)
        elif option == 'vgg11':
            vgg16 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pret else None)
        elif option == 'vgg13':
            vgg16 = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1 if pret else None)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1 if pret else None)
        elif option == "vgg16":
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pret else None)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pret else None)
        elif option == "vgg19":
            vgg16 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pret else None)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if pret else None)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x, source=True,target=False):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)
