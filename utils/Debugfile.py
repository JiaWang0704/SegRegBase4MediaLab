# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
""" Construct Matrix with from (angle, scale_x, scale_y, center_x, center_y) """
# **********************************

from dataset import read_img2gray
import matplotlib.pyplot as plt
from torchvision import transforms
from losses import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def construct_M(angle, scale_x, scale_y, center_x, center_y):
    alpha = torch.cos(angle)
    beta = torch.sin(angle)
    print(alpha, beta)
    tx = center_x
    ty = center_y
    tmp0 = torch.cat((scale_x * alpha, beta), 1)
    tmp1 = torch.cat((-beta, scale_y * alpha), 1)
    print(tmp0, tmp1)
    theta = torch.cat((tmp0, tmp1), 0)
    t = torch.cat((tx, ty), 0)
    matrix = torch.cat((theta, t), 1)
    return theta, matrix


degree = -1.0
angle, scale_x, scale_y, center_x, center_y = torch.Tensor([[degree]]), \
                                              torch.Tensor([[0.8]]), torch.Tensor([[1.3]]), \
                                              torch.Tensor([[0.4]]), torch.Tensor([[0.3]])
theta, matrix = construct_M(angle, scale_x, scale_y, center_x, center_y)
print(matrix)


# Det loss check
# aff = theta
# aff = torch.Tensor(aff)
det_value = torch.det(theta)
print(det_value.item())
det_loss = torch.mean((det_value - 1.0) ** 2)
print(det_loss.item())

# load image
src = read_img2gray('COAD_09/scale-25pc/S1.jpg')
src = transforms.ToTensor()(src).float()
grid = F.affine_grid(matrix.unsqueeze(0), src.unsqueeze(0).size(), align_corners=True)
warp = F.grid_sample(src.unsqueeze(0), grid, padding_mode='border', align_corners=True)

# # Ncc loss check
# new_img_torch = warp[0]
# ncc0 = similarity_loss(src.unsqueeze(0), new_img_torch.unsqueeze(0))
# ncc1 = similarity_loss(src.unsqueeze(0), src.unsqueeze(0))
# print(ncc0.item())
# print(ncc1.item())

# show image
fig, axes = plt.subplots(1, 2, figsize=(6, 6))
ax0, ax1 = axes.ravel()

ax0.imshow(src.numpy().transpose(1,2,0))
ax0.set_title("Source image")

ax1.imshow(warp[0,:,:,:].numpy().transpose(1,2,0))
ax1.set_title("Warped imge")

for ax in axes.ravel():
    ax.axis('off')
fig.tight_layout()
plt.show()