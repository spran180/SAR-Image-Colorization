import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')

class SarOpticalDataset(torch.utils.data.Dataset):
    
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
    transforms.ToTensor()
])


sar_path = '/kaggle/input/sar-dataset/QXSLAB_SAROPT/sar_256_oc_0.2'
opt_path = '/kaggle/input/sar-dataset/QXSLAB_SAROPT/opt_256_oc_0.2'

dataset = SarOpticalDataset(sar_path=sar_path, opt_path=opt_path, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class BasicUNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channel, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channel, kernel_size=5, padding=2)
        ])

        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(l(x))

        return x

def print_image(x, y):
    grid_img = torchvision.utils.make_grid(x)

    # Convert tensor to a NumPy array and adjust the range if necessary
    np_img = grid_img.numpy().transpose((1, 2, 0)).clip(0, 1)
    
    # Display the image with Matplotlib
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

    grid_img = torchvision.utils.make_grid(y)
    
    # Convert tensor to NumPy array and clip values
    np_img = grid_img.numpy().transpose((1, 2, 0)).clip(0, 1)
    
    # Display predictions with Matplotlib
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()
    
    # Model predictions
    with torch.no_grad():
        preds = net(x.to(device), 0).sample.detach().cpu()
    
    # Create a grid from predictions
    grid_img = torchvision.utils.make_grid(preds)
    
    # Convert tensor to NumPy array and clip values
    np_img = grid_img.numpy().transpose((1, 2, 0)).clip(0, 1)
    
    # Display predictions with Matplotlib
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()


batch_size = 64
train_dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epoch = 3

net = model = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
).to(device)

loss_fun = nn.MSELoss()

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = []

for epoch in range(n_epoch):
    for x, y in train_dataLoader:

        x = x.to(device)
        y = y.to(device)
        # noise_amount = torch.rand(x.shape[0]).to(device)
        # noisy_x = corrupt(x, noise_amount)

        pred = net(x, 0).sample
        loss = loss_fun(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
    avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

plt.plot(losses)
plt.ylim(0, 0.1)



torch.save(net.state_dict(), 'model_parameters_3.pth')

net.load_state_dict(torch.load('/kaggle/input/diffusionmodel/pytorch/default/1/model_parameters_2.pth')) 
net.eval()

Code for plot actual images and prewdiceted images


x, y = next(iter(train_dataLoader))
x = x[:8]  # Select the first 8 images

# Create a grid from the batch of images
print_image(x[:8], y[:8])

