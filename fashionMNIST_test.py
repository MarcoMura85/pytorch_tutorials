import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

torch.set_grad_enabled(False)
network = networkClass.Network()
network.load_state_dict(torch.load(".\\trained\\batch_size=10 lr=0.001 shuffle=True epochs=10.pt"))

batch_size = 100
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

images, labels = next(iter(test_loader))

total_loss = 0
total_correct = 0


prediction_loader = torch.utils.data.DataLoader(test_set, batch_size=10000)
test_preds = util.get_all_preds(network, prediction_loader)
preds_correct = util.get_num_correct(test_preds, test_set.targets)

util.print_training_results('N/A', preds_correct, 'N/A', len(test_loader))

cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))

names = (
    'T-shirt/top'
    ,'Trouser'
    ,'Pullover'
    ,'Dress'
    ,'Coat'
    ,'Sandal'
    ,'Shirt'
    ,'Sneaker'
    ,'Bag'
    ,'Ankle boot'
)

plt.figure(figsize=(10, 10))
util.plot_confusion_matrix(cm, names)
