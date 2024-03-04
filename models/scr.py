import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


class SCR(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        #self.dropout = nn.Dropout(0.5)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]),)

    def forward(self, x, training=True):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)

        x = self.conv1x1_in(x)   # [80, 640, hw, 25] -> [80, 64, HW, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        return x

#"""
#first-这是最原始的SCR模块
class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)#
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        #print("x.shape_before",x.shape)#torch.Size([80, 640, 5, 5])
        x = self.relu(x)#变为非线性
        #print("x.shape_after",x.shape)#torch.Size([80, 640, 5, 5])
        x = F.normalize(x, dim=1, p=2)#归一化处理
        #print("x.shape",x.shape)#torch.Size([80, 640, 5, 5])  torch.Size([8, 640, 5, 5])两个交替出现
        identity = x
        #print("identity.shape",identity.shape)#torch.Size([80, 640, 5, 5])  torch.Size([8, 640, 5, 5])两个交替出现

        x = self.unfold(x)  # b, cuv, h, w 展平操作，即论文中提到的领域
        #print("x.shape",x.shape)#torch.Size([80, 16000, 25])  torch.Size([8, 16000, 25])两个交替出现
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)#张量的形状重塑操作
        #print("x.shape",x.shape)#torch.Size([80, 640, 5, 5, 5, 5])  torch.Size([8, 640, 5, 5, 5, 5])两个交替出现
        x = x * identity.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        #print("x.shape",x.shape)#torch.Size([80, 640, 5, 5, 5, 5])  torch.Size([8, 640, 5, 5, 5, 5])两个交替出现
        #执行两次 unsqueeze(2)，相当于在索引位置 2 和 3 上分别插入了两个新的维度，从而使得张量的维度数增加了 2
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v维度重排
        #print("x.shape",x.shape)#torch.Size([80, 640, 5, 5, 5, 5])  torch.Size([8, 640, 5, 5, 5, 5])两个交替出现
        return x
#"""



"""
#second-这是更改为kronnecker product的SCR模块
class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)

        # Compute Kronecker product
        identity_expanded = identity.unsqueeze(2).unsqueeze(3)
        identity_expanded = identity_expanded.expand(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity_expanded

        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        return x
"""



"""
#third-这是更改为kronnecker product+PCA的SCR模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2, pca_components=128):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pca_components = pca_components
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pca = PCA(n_components=pca_components)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        # Reshape input tensor for PCA
        x_pca = x.permute(0, 2, 3, 1).contiguous().view(-1, c)

        # Apply PCA for dimensionality reduction
        x_pca = self.pca.fit_transform(x_pca.detach().cpu().numpy())
        x_pca = torch.from_numpy(x_pca).to(x.device)
        x_pca = x_pca.view(b, h, w, self.pca_components).permute(0, 3, 1, 2)

        # Compute self-correlation with reduced dimensionality
        x_pca = self.relu(x_pca)
        x_pca = F.normalize(x_pca, dim=1, p=2)
        identity_pca = x_pca

        x_pca = self.unfold(x_pca)
        x_pca = x_pca.view(b, self.pca_components, self.kernel_size[0], self.kernel_size[1], h, w)
        identity_expanded = identity_pca.unsqueeze(2).unsqueeze(3)
        identity_expanded = identity_expanded.expand(b, self.pca_components, self.kernel_size[0], self.kernel_size[1], h, w)
        x_pca = x_pca * identity_expanded
        x_pca = x_pca.permute(0, 1, 4, 5, 2, 3).contiguous()

        return x_pca
"""



"""
#forth-这是更改为高斯平均池化的SCR模块
class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        # 高斯平均池化操作
        x = F.avg_pool2d(x, kernel_size=self.kernel_size, padding=self.padding)

        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()

        return x
"""






















