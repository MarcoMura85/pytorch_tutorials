

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

