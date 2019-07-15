import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import networkClass

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# train_loader = torch.utils.data.DataLoader(
#     train_set
#     , batch_size=100
# )
# print(len(train_set))
# print(train_set.targets)
# print(train_set.targets.bincount())

network = networkClass.Network()

# sample = next(iter(train_set))
# image,label = sample
# #print(label)
# image.unsqueeze(0).shape #transforming it in a batch
#
# pred = network(image.unsqueeze(0))
#
# print(pred.argmax(dim=1))
# print(F.softmax(pred, dim=1))

# plt.imshow(image.squeeze(), cmap='gray')
# print('label:', label)
# plt.show()

# optimizer = optim.Adam(network.parameters(), lr=0.01)
#
# batch = next(iter(train_loader))
# images, labels = batch
#
# preds = network(images)
# loss = F.cross_entropy(preds, labels) # Calculating the loss

# print(preds)
# print(preds.argmax(dim=1))
# print(preds.argmax(dim=1).eq(labels))

# loss.backward() # Calculate Gradients
# optimizer.step() # Update Weights
#
# print('loss1:', loss.item())
# print(get_num_correct(preds, labels))
#
# preds = network(images)
# loss = F.cross_entropy(preds, labels)
# print('loss2:', loss.item())
# print(get_num_correct(preds, labels))

#
# grid = torchvision.utils.make_grid(images, nrow=10)
# plt.figure(figsize=(15, 15))
# plt.imshow(np.transpose(grid, (1, 2, 0)))
# print(labels)
# plt.show()


train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

for epoch in range(5):

    for batch in train_loader:  # Get Batch
        images, labels = batch

        preds = network(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss

        print('loss:', loss.item())
        #print(get_num_correct(preds, labels))

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch:", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )

    print(total_correct / len(train_set))