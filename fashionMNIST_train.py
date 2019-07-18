import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from itertools import product

import torchvision
import torchvision.transforms as transforms
import networkClass
import utils as util

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# batch_size_list = [10, 100, 1000]
# lr_list = [.01, .001]
# shuffle_list = [True, False]

use_tensorboard = False

batch_size_list = [10]
lr_list = [.001]
shuffle_list = [True]
epoch_length = 10

param_values = util.get_hyperparams_values(lr_list, batch_size_list, shuffle_list)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for lr, batch_size, shuffle in product(*param_values):

    network = networkClass.Network()
    #network.to(device)
    #torch.backends.cudnn.benchmark=True
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
    if use_tensorboard:
        tb = SummaryWriter(comment=comment)
        tb.add_image('images', grid)
        tb.add_graph(network, images)

    for epoch in range(epoch_length):

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

            total_loss += loss.item() * batch_size
            total_correct += util.get_num_correct(preds, labels)

        if use_tensorboard:
            util.add_scalars_to_tensorboard(tb, epoch, total_correct, total_loss, len(train_set))
            util.add_histograms_to_tensorboard(network, tb, epoch)

        util.print_training_results(epoch, total_correct, total_loss, len(train_set))

    torch.save(network.state_dict(), ".\\trained\\"+comment.strip()+" epochs="+str(epoch_length)+".pt")

if use_tensorboard:
    tb.close()
