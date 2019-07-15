import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import networkClass

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
                                           , batch_size=10
                                           )
# print(len(train_set))
# print(train_set.targets)
# print(train_set.targets.bincount())

sample = next(iter(train_set))
image,label = sample
# print(label)
#
# plt.imshow(image.squeeze(), cmap='gray')
# print('label:', label)
# plt.show()

batch = next(iter(train_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print(labels)
plt.show()