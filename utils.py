import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def get_hyperparams_values(lr_list, batch_size_list, shuffle_list):
    parameters = dict(
        lr=lr_list
        , batch_size=batch_size_list
        , shuffle=shuffle_list
    )
    return [v for v in parameters.values()]

def add_scalars_to_tensorboard(tb, epoch, total_correct, total_loss, len_train_set):
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len_train_set, epoch)


def add_histograms_to_tensorboard(network, tb, epoch):
    for name, param in network.named_parameters():
        tb.add_histogram(name, param, epoch)
        tb.add_histogram(f'{name}.grad', param.grad, epoch)


def print_training_results(epoch, total_correct, total_loss, len_train_set):
    print(
        "epoch:", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )
    print(
        "Accurancy: ", total_correct / len_train_set
    )

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.rainbow):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()