import torch
import torch.nn as nn
import torch.nn.functional as F
from grl import GradientReverseLayer


class DGImgHead(nn.Module):
    def __init__(self,dim,num_domains):
        super(DGImgHead,self).__init__()
        self.dim=dim 
        self.num_domains = num_domains
        self.gradient_reverse_layer = GradientReverseLayer()
        self.Conv1 = nn.Conv2d(self.dim, 256, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=1)

        for l in [self.Conv1, self.Conv2, self.Conv3, self.Conv4]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)


        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

        nn.init.normal_(self.classifier.weight, std=0.05)
        nn.init.constant_(self.classifier.bias, 0)
        
        
    def forward(self,x):
        x=self.gradient_reverse_layer(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.fc(x))
        x=self.classifier(x)
        
        return self.softmax(x)


class DGInsHead(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DGInsHead, self).__init__()
        self.gradient_reverse_layer = GradientReverseLayer()
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
        x=self.gradient_reverse_layer(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5)

        x = self.classifier(x)

        return self.softmax(x)