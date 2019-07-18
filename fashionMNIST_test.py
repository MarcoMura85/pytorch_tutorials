import torch
import torchvision
import torchvision.transforms as transforms
import networkClass
import utils as util


test_set = torchvision.datasets.FashionMNIST(
    root='./dataTest'
    , train=False
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

print(len(test_set))