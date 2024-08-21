import matplotlib.pyplot as plt
import numpy as np

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
print("all Good till now")

class SarOpticalDataset(Dataset):
    
    def __init__(self, sar_path, opt_path, transform=None):
        self.sar_dirs = sar_path
        self.opt_dirs = opt_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(sar_path) if os.path.isfile(os.path.join(sar_path, f))]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        sar_image_path = os.path.join(self.sar_dirs, self.image_filenames[idx])
        sar_image = Image.open(sar_image_path)
        
        optical_image_path = os.path.join(self.opt_dirs, self.image_filenames[idx])
        optical_image = Image.open(optical_image_path)
        
        if(self.transform):
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)
            
        return sar_image, optical_image
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming SAR images are grayscale
])


sar_path = '/media/pranav-sharma/New Volume/QXSLAB_SAROPT/sar_256_oc_0.2'
opt_path = '/media/pranav-sharma/New Volume/QXSLAB_SAROPT/opt_256_oc_0.2'

dataset = SarOpticalDataset(sar_path=sar_path, opt_path=opt_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

dataset.__getitem__(1)[0].shape

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma*out + x
        return out

class SelfAttention(nn.Module):

    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.to_chunks = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, n, d = x.size()
        h =self.heads
        qkv = self.to_chunks(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)

        wei = (q @ k.transpose(-2, -1)) * (d ** -0.5)
        wei = F.softmax(wei, dim=-1)
        out = (wei @ v).transpose(1, 2).reshape(b, n, d)

        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, features, in_channels, out_channels):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.self_attention = SelfAttention(features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        x = self.self_attention(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        x = self.self_attention(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        return x
    
class ColorizeSARimage(nn.Module):
    def __init__(self, features=64):
        super(ColorizeSARimage, self).__init__()
        
        self.encoder = Encoder(features, 1, features)
        self.decoder = Decoder(features, features)
        self.conv = nn.Conv2d(features, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.tanh(self.conv(x))
        return x
    
model = ColorizeSARimage(features=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device) 

for epoch in range(10):
    for i, data in enumerate(dataloader):
        
        sar, optical = data
        sar, optical = sar.to(device), optical.to(device)
        print(sar.shape)
        optimizer.zero_grad()
        output = model(sar)
        loss = criterion(output, optical)
        loss.backward()
        optimizer.step()
        
        if(i % 10 == 0):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, i+1, len(dataset)//16, loss.item()))

