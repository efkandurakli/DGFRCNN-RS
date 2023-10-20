import torch
import torch.nn as nn
import torch.nn.functional as F
from fasterrcnn import fasterrcnn_resnet50_fpn
from dg_heads import DGImgHead, DGInsHead

class DGFasterRCNN(nn.Module):
    def __init__(
            self, 
            num_classes, 
            num_domains,
            weights=None, 
            weights_backbone=None, 
            img_dg = False,
            ins_dg = False,
            **kwargs
    ):
        super().__init__()
        self.detector = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone, num_classes=num_classes, **kwargs)
        self.num_domains = num_domains
        self.img_dg = img_dg
        self.ins_dg = ins_dg

        if self.img_dg:
            self.detector.backbone.register_forward_hook(self.store_img_features)
            self.imghead = DGImgHead(256, self.num_domains)

        if self.ins_dg:
            self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)
            self.inshead = DGInsHead(1024, self.num_domains)

    def store_img_features(self, module, input, output):
       self.img_features = output

    def store_ins_features(self, module, input, output):
        self.ins_domains = input[1]
        self.ins_features = output


    def forward(self, images, targets=None):
        if targets is not None:
            losses = self.detector(images, targets)

            if self.img_dg:
                img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)
                img_domain_logits = self.imghead(self.img_features["0"])
                img_dg_loss = F.cross_entropy(img_domain_logits, img_domain_labels)
                losses.update({"img_dg_loss": img_dg_loss})

            if self.ins_dg:
                ins_domain_logits = self.inshead(self.ins_features)
                ins_dg_loss = F.cross_entropy(ins_domain_logits, self.ins_domains)
                losses.update({"ins_dg_loss": ins_dg_loss})


            return losses

        else:
            return self.detector(images)
    
