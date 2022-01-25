import torch
#from torch._C import float32
import torch.nn as nn
from torch import Tensor
import torchvision.models as models

from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

import config


class ResConvBlock(nn.Module):
    """
    Applies the residual convolution layers while maintaining the number of channels and image size to be the same

    Args:
        channels (int) : Number of channels in the input image 
    """

    def __init__(self, channels: int) -> None:
        super(ResConvBlock, self).__init__()
        self.r_c_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        original = x
        output = self.r_c_block(x)
        output = output + original

        return output


class Discriminator1(nn.Module):
    def __init__(self) -> None:
        super(Discriminator1, self).__init__()
        self.features = nn.Sequential(
            # Input size of images = (3) x 256 x 256
            nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias = True),
            nn.LeakyReLU(0.2, True),

            # Size = (64) x 127 x 127
            nn.Conv2d(64, 64, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(128),

            # Size = (128) x 62 x 62
            nn.Conv2d(128, 128, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # Size = (256) x 29 x 29
            nn.Conv2d(256, 256, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),

            # Size = (512) x 13 x 13
            nn.Conv2d(512, 512, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, bias = False),
            nn.Conv2d(512, 1024, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),

            # Size = (1024) x 5 x 5
            nn.Conv2d(1024, 1024, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, bias = False),
            nn.Conv2d(1024, 1024, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Discriminator2(nn.Module):
    def __init__(self) -> None:
        super(Discriminator2, self).__init__()
        self.features = nn.Sequential(
            # Input size of images = (3) x 512 x 512
            nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias = True),
            nn.LeakyReLU(0.2, True),

            # Size = (64) x 255 x 255
            nn.Conv2d(64, 64, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # Size = (128) x 126 x 126
            nn.Conv2d(128, 128, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # Size = (128) x 61 x 61
            nn.Conv2d(128, 128, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # Size = (256) x 29 x 29
            nn.Conv2d(256, 256, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # Size = (512) x 13 x 13
            nn.Conv2d(512, 512, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, bias = False),
            nn.Conv2d(512, 1024, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),

            # Size = (1024) x 5 x 5
            nn.Conv2d(1024, 1024, (4, 4), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, bias = False),
            nn.Conv2d(1024, 1024, (4, 4), (2, 2), (1, 1), bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        # First Conv Layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(64)
        )

        # 16 Residual Blocks
        blocks = []
        for _ in range(16):
            blocks.append(ResConvBlock(64))
        self.blocks = nn.Sequential(*blocks)

        # Second Conv Layer
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias = False),
            nn.BatchNorm2d(64)
        )

        # Upsampling the Image
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # First output for Discriminator 1 - D1
        self.output1 = nn.Conv2d(256, 3, (3, 3), (1, 1), (1, 1))

        # Second output for Discriminator 2 - D2
        self.output2 = nn.Conv2d(256, 3, (3, 3), (1, 1), (1, 1))

        # Initialize Weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_pass(x)

    def forward_pass(self, x: Tensor) -> Tensor:
        o1 = self.conv_block1(x)
        o2 = self.blocks(o1)
        o3 = self.conv_block2(o2)
        o4 = o1 + o3

        out1 = self.upsampling(o4)
        out1 = self.output1(out1) # Goes to first discriminator D1

        out2 = self.upsampling(out1)
        out2 = self.output2(out2) # Goes to second discriminator D2
    
        return out2

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1


class YOLOv5(nn.Module):
    def __init__(self) -> None:
        super(YOLOv5, self).__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape = False, pretrained = False, classes = 4)
    
    def forward(self, img):
        results = self.model(img)
        return results


class ModelEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC) -> None:
        super(ModelEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

    def forward(self, low_res, high_res):
        sr_img = self.modelA(low_res)
        sr_disc = self.modelB(sr_img)
        hr_disc = self.modelB(high_res)
        detection = self.modelC(sr_img)
        return sr_img, sr_disc, hr_disc, detection

class mseLoss(nn.Module):
    def __init__(self) -> None:
        super(mseLoss, self).__init__()
    
    def mse_forward(self, sr: Tensor, hr: Tensor):
        loss = nn.MSELoss()
        mse_loss = loss(sr, hr)

        return mse_loss


class Perceptual_Loss(nn.Module):
    def __init__(self) -> None:
        super(Perceptual_Loss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor):
        # Standardized operations.
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        # Find the feature map difference between the two images.
        loss = mseLoss.mse_forward(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss


class Adversarial_Loss(nn.Module):
    def __init__(self) -> None:
        super(Adversarial_Loss, self).__init__()
    
    def forward(self, sr: Tensor, hr: Tensor, discriminator = Discriminator2):
        hr_loss = torch.log(discriminator.forward(hr))
        sr_loss = torch.log(1 - discriminator.forward(sr))

        total_loss = torch.sum(hr_loss, sr_loss)

        return total_loss


class Detection_Loss(nn.Module):
    def __init__(self) -> None:
        super(Detection_Loss, self).__init__()
        # Loss Coefficients
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, out, label, obj = True):
        # Predictions by yolov5
        conf_pred = out[0]
        box_pred = out[1:5]
        classes_pred = out[5:] # Confidence values for all the classes
        class_pred = torch.argmax(classes_pred) # Class with highest conf score
        x_pred = box_pred[0]
        y_pred = box_pred[1]
        w_pred = box_pred[2]
        h_pred = box_pred[3]

        # Ground truth values     
        box_true = label[1:5]
        class_true = label[0]
        x_true = box_true[0]
        y_true = box_true[1]
        w_true = box_true[2]
        h_true = box_true[3]

        # Loss Functions
        loss_obj = -(obj * torch.log(conf_pred))
        loss_noobj = -((1 - obj) * torch.log(1 - conf_pred))
        loss_center = 0
        loss_size = 0
        loss_class = 0
        #loss_wrong_class = 0

        if(obj):
            loss_center += ((x_pred - x_true) ** 2) + ((y_pred - y_true) ** 2)
            loss_size += ((torch.sqrt(w_pred) - torch.sqrt(w_true)) ** 2) + ((torch.sqrt(h_pred) - torch.sqrt(h_true)) ** 2)
            loss_class += -torch.log(classes_pred[class_true])
            #if(class_true != class_pred):
            #    loss_wrong_class += -torch.log(1 - classes_pred[class_pred])

        total_loss = (self.lambda_coord * loss_center) + (self.lambda_coord * loss_size) + loss_obj + (self.lambda_noobj * loss_noobj) + loss_class
        #total_loss = float32(total_loss)

        return total_loss


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.alpha = config.ALPHA
        self.beta = config.BETA
        self.gamma = config.GAMMA

    def forward(self, mse_loss, perc_loss, adv_loss, det_loss):
        total_loss = mse_loss + (self.alpha * perc_loss) + (self.beta * adv_loss) + (self.gamma * det_loss)

        return total_loss