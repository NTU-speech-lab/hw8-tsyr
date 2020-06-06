import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 270)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])


source_dataset = ImageFolder('/tmp3/real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('/tmp3/real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=512, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=512, shuffle=True)

class FeatureExtractor(nn.Module):
    
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


feature_extractor = FeatureExtractor().cuda()
feature_extractor.load_state_dict(torch.load('./models/extractor_model.bin'))


def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='with DaNN')
    if savefig is not None:
        plt.savefig(savefig)
    # plt.show()
    return

domain = []
X_embedded = []
label_predictor.eval()
feature_extractor.eval()
for i, ((source_data, _0), (target_data, _1)) in enumerate(zip(source_dataloader, target_dataloader)):
    if i > 0:
        break
    source_data = source_data.cuda()
    target_data = target_data.cuda()
    mixed_data = torch.cat([source_data, target_data], dim=0)

    ft = feature_extractor(mixed_data)
    x = TSNE(n_components=2).fit_transform(ft)

    for q in x:
        X_embedded.append(q)
    for i in range(source_data.shape[0]):
        domain.append(0)
    for i in range(target_data.shape[0]):
        domain.append(1)

plot_scatter(X_embedded, domain, savefig='./with_DaNN_scatter.png')