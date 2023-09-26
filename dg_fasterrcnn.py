import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from gradient_scalar_layer import GradientScalarLayer

DG_IMG_GRL_WEIGHT = 0.1
DG_INS_GRL_WEIGHT = 0.1

DG_IMG_LOSS_WEIGHT = 0
DG_INS_LOSS_WEIGHT = 0
DG_CST_LOSS_WEIGHT = 0

class DGImgHead(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DGImgHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        self.fc1 = nn.Linear(53294, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_domains)
        self.softmax = nn.Softmax(dim=1)

        for l in [self.conv1, self.conv2]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

        for l in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.classifier.weight, std=0.05)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1(feature))
            img_features.append(self.conv2(t))

        img_features_flattened = []
        for img_feature in img_features:
            N = img_feature.shape[0]
            img_feature = img_feature.permute(0, 2, 3, 1)
            img_feature = img_feature.reshape(N, -1)
            img_features_flattened.append(img_feature)

        img_features_flattened = torch.cat(img_features_flattened, dim=1)

        img_features_flattened = F.relu(self.fc1(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = F.relu(self.fc2(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = F.relu(self.fc3(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = self.classifier(img_features_flattened)


        return self.softmax(img_features_flattened)
    
class DGInsHead(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DGInsHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_domains)
        self.softmax = nn.Softmax(dim=1)

        for l in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.05)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5)

        x = self.classifier(x)

        return self.softmax(x)


class DGFasterRCNN(nn.Module):
    
    def __init__(self, num_classes, num_domains=10, **kwargs):
        super().__init__()
        self.detector = fasterrcnn_resnet50_fpn(num_classes=num_classes, **kwargs)

        self.detector.backbone.register_forward_hook(self.store_img_features)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)

        self.grl_img = GradientScalarLayer(-1.0*DG_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*DG_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*DG_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*DG_INS_GRL_WEIGHT)

        self.imghead = DGImgHead(256, num_domains)
        self.inshead = DGInsHead(1024, num_domains)
        

    def store_img_features(self, module, input, output):
       self.img_features = output
      
            
    def store_ins_features(self, module, input, output):
        self.box_features = output

    def forward(self, images, targets=None):
        if targets is not None:
            loss_dict = self.detector(images, targets)
            
            img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)

            batch_per_image = self.box_features.shape[0] // len(targets)

            ins_domain_labels = []
            for target in targets:
                ins_domain_labels.append(target["domain"].repeat(batch_per_image))

            ins_domain_labels = torch.cat(ins_domain_labels, dim=0)

            img_grl_fea = [self.grl_img(self.img_features[fea]) for fea in self.img_features]
            ins_grl_fea = self.grl_ins(self.box_features)
            img_grl_consist_fea = [self.grl_img_consist(self.img_features[fea]) for fea in self.img_features]
            ins_grl_consist_fea = self.grl_ins_consist(self.box_features)

            img_domain_logits = self.imghead(img_grl_fea)

            img_loss = F.cross_entropy(img_domain_logits, img_domain_labels)


            ins_domain_logits = self.inshead(ins_grl_fea)

            ins_loss = F.cross_entropy(ins_domain_logits, ins_domain_labels)


            dg_img_consist_features = self.imghead(img_grl_consist_fea)
            dg_ins_consist_features = self.inshead(ins_grl_consist_fea)

            loss_dict.update({"img_loss": img_loss, "ins_loss": ins_loss})

            return loss_dict
        else:
            return self.detector(images)