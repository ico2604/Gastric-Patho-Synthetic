import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class GastricMTLModel(nn.Module):
    def __init__(self, n_seg_classes=5, n_cls_classes=4):
        super().__init__()
        # 공유 인코더 및 세그먼테이션 디코더
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_seg_classes
        )
        
        # 분류 헤드: 인코더의 마지막 특징(2048ch) 활용
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_cls_classes)
        )

    def forward(self, x):
        # 1. Shared Encoder
        features = self.unet.encoder(x)
        
        # 2. Segmentation Branch
        decoder_output = self.unet.decoder(features)
        seg_out = self.unet.segmentation_head(decoder_output)
        
        # 3. Classification Branch
        cls_feat = self.avgpool(features[-1]) # 마지막 레이어 (Batch, 2048, 1, 1)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.classifier(cls_feat)
        
        return seg_out, cls_out

    def freeze_encoder(self):
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True