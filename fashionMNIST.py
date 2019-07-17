import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from itertools import product

import torchvision
import torchvision.transforms as transforms
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

network = networkClass.Network()
torch.backends.cudnn.benchmark=True
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb = SummaryWriter()
tb.add_image('images', grid)
tb.add_graph(network, images)

for epoch in range(10):

    print("starting epoch:", epoch)

    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # Get Batch
        images, labels = batch

        preds = network(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss

        #print('loss:', loss.item())
        #print(get_num_correct(preds, labels))

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    for name, param in network.named_parameters():
        tb.add_histogram(name, param, epoch)
        tb.add_histogram(f'{name}.grad', param.grad, epoch)

    print(
        "epoch:", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )

print(total_correct / len(train_set))
tb.close()