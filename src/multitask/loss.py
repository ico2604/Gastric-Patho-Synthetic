import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedMTLLoss(nn.Module):
    def __init__(self, l_seg=1.0, l_cls=0.5):
        super().__init__()
        self.l_seg = l_seg
        self.l_cls = l_cls
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.ce_seg = nn.CrossEntropyLoss()
        self.ce_cls = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_gt, cls_pred, cls_gt):
        # Segmentation Loss
        loss_seg = self.dice(seg_pred, seg_gt) + self.ce_seg(seg_pred, seg_gt)
        # Classification Loss
        loss_cls = self.ce_cls(cls_pred, cls_gt)
        
        total = (self.l_seg * loss_seg) + (self.l_cls * loss_cls)
        return total, loss_seg, loss_cls