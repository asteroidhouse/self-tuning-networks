import os
import sys
import csv
import ipdb
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision.utils import make_grid
from torchvision import datasets, transforms

file_path = os.path.realpath(__file__)
trim_length = len(file_path.split('/')[-1])
file_dir = file_path[:-trim_length]
sys.path.insert(0, file_dir + '/util')
from cutout import Cutout

sys.path.insert(0, file_dir + '/models')
from small import SmallCNN
from alexnet import AlexNet

sys.path.insert(0, file_dir + '/datasets')
from loaders import create_loaders

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from logger import Logger


def cnn_val_loss(config={}, reporter=None, callback=None, return_all=False):
    print("Starting cnn_val_loss...")

    ###############################################################################
    # Arguments
    ###############################################################################
    model_options = ['generic', 'small', 'alexnet']
    dataset_options = ['cifar10', 'cifar100']

    ## Tuning parameters: all of the dropouts
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
    parser.add_argument('--model', '-a', default='alexnet', choices=model_options)
    #### Optimization hyperparameters
    parser.add_argument('--batch_size', '-bsz', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--total_epochs', '-totep', type=int, default=8000,
                        help='number of epochs to train (default: 8000)')
    parser.add_argument('--train_lr', '-tlr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', '-mom', type=float, default=0.9,
                        help='learning rate')

    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='Factor by which to multiply the learning rate.')
    parser.add_argument('--nonmono', type=int, default=60,
                        help='Number of epochs for nonmonotonic lr decay')
    parser.add_argument('--patience', type=int, default=75,
                        help='How long to wait for the val loss to improve before early stopping.')

    #### Regularization hyperparameters
    parser.add_argument('--indropout', type=float, default=0. if 'indropout' not in config else config['indropout'],
                        help='starting dropout rate')
    # Dropout hyperparameters
    parser.add_argument('--dropout', '-drop', type=float, default=-1 if 'dropout' not in config else config['dropout'],
                        help='starting dropout rate shared between layers- active if > 0')
    parser.add_argument('--dropout0', type=float, default=0. if 'dropout0' not in config else config['dropout0'],
                        help='starting dropout rate for layer 0')
    parser.add_argument('--dropout1', type=float, default=0. if 'dropout1' not in config else config['dropout1'],
                        help='starting dropout rate for layer 1')
    parser.add_argument('--dropout2', type=float, default=0. if 'dropout2' not in config else config['dropout2'],
                        help='starting dropout rate for layer 2')
    parser.add_argument('--dropout3', type=float, default=0. if 'dropout3' not in config else config['dropout3'],
                        help='starting dropout rate for layer 3')
    parser.add_argument('--dropout4', type=float, default=0. if 'dropout4' not in config else config['dropout4'],
                        help='starting dropout rate for layer 4')
    # Fully connected layer dropout hyperparameters (used in AlexNet)
    parser.add_argument('--fc_dropout', '-fcdrop', type=float, default=-1,
                        help='starting dropout rate shared between layers- active if > 0')
    parser.add_argument('--fc_dropout0', '-fcdrop0', type=float, default=0. if 'fc_dropout0' not in config else config['fc_dropout0'],
                        help='starting dropout rate for layer 0')
    parser.add_argument('--fc_dropout1', '-fcdrop1', type=float, default=0. if 'fc_dropout1' not in config else config['fc_dropout1'],
                        help='starting dropout rate for layer 1')
    # Other regularization hyperparameters
    parser.add_argument('--inscale', '-iscl', type=float, default=0 if 'inscale' not in config else config['inscale'],
                        help='defines input scaling factor')
    parser.add_argument('--hue', type=float, default=0. if 'hue' not in config else config['hue'],
                        help='hue jitter rate')
    parser.add_argument('--brightness', type=float, default=0. if 'brightness' not in config else config['brightness'],
                        help='brightness jitter rate')
    parser.add_argument('--saturation', type=float, default=0. if 'saturation' not in config else config['saturation'],
                        help='saturation jitter rate')
    parser.add_argument('--contrast', type=float, default=0. if 'contrast' not in config else config['contrast'],
                        help='contrast jitter rate')
    parser.add_argument('--data_augmentation', '-daug', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--cutholes', '-cuth', type=int, default=-1 if 'cutholes' not in config else int(round(config['cutholes'])),
                        help='number of holes to cut out from image in cutout- active if >0')
    parser.add_argument('--cutlength', '-cutl', type=int, default=-1 if 'cutlength' not in config else int(round(config['cutlength'])), help='length of the holes in cutout- active if >0')
    ##### Miscellaneous hyperparameters
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save current run')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--percent_valid', type=float, default=0.2,
                        help='percentage of training dataset to be used as a validation set')
    parser.add_argument('--logdir', default='logs',
                        help='directory of regNet to save to')
    parser.add_argument('--save_dir', default=config['save_dir'],
                        help='subdirectory of logdir/savedir to save in (default changes to date/time)')

    args, unknown = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)
    sys.stdout.flush()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ###############################################################################
    # Saving
    ###############################################################################
    train_labels = ('global_step', 'time', 'loss', 'acc')
    valid_labels = ('global_step', 'time', 'loss', 'acc')
    test_labels = ('global_step', 'time', 'loss', 'acc')

    label_dict = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}

    ###############################################################################
    # Data Loading/Processing
    ###############################################################################
    train_loader, valid_loader, test_loader = create_loaders(args)

    ###############################################################################
    # Model/Optimizer
    ###############################################################################
    num_classes = 10

    if args.model == 'small':
        if args.dropout > 0:
            dropRates = [args.dropout for _ in range(3)]
        else:
            dropRates = [args.dropout0, args.dropout1, args.dropout2]
        cnn = SmallCNN(num_classes=num_classes, dropRates=dropRates)

    elif args.model == 'alexnet':
        if args.dropout > 0:
            dropRates = [args.dropout for _ in range(5)]
        else:
            dropRates = [args.dropout0, args.dropout1, args.dropout2, args.dropout3, args.dropout4]
        if args.fc_dropout > 0:
            fc_dropRates = [args.fc_dropout for _ in range(2)]
        else:
            fc_dropRates = [args.fc_dropout0, args.fc_dropout1]

        cnn = AlexNet(num_classes=num_classes, dropRates=dropRates,
                      fc_dropRates=fc_dropRates, filters=[64, 192, 384, 256, 256])

    #### Regularization hyperparameters
    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.train_lr, momentum=args.momentum)

    ###############################################################################
    # Training/Evaluation
    ###############################################################################
    def evaluate(loader):
        """Returns the loss and accuracy on the entire validation/test set.

        Arguments:
        loader -- a DataLoader wrapping around the validation/test set
        """
        cnn.eval()    # Change model to 'eval' mode.
        correct = total = loss = 0.
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                pred = cnn(images)
                loss += F.cross_entropy(pred, labels, reduction='sum').item()
                hard_pred = torch.max(pred, 1)[1]
                total += labels.size(0)
                correct += (hard_pred == labels).sum().item()

        accuracy = correct / total
        mean_loss = loss / total
        cnn.train()
        return mean_loss, accuracy

    epoch = 1
    global_step = 0
    patience_elapsed = 0
    stored_loss = 1e8
    best_val_loss = []
    start_time = time.time()

    while patience_elapsed < args.patience:

        running_xentropy = correct = total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images, labels = images.to(device), labels.to(device)
            if args.inscale > 0:
                noise = torch.rand(images.size(0), device=device)
                scaled_noise = ((1 + args.inscale) - (1 / (1 + args.inscale))) * noise + (1 / (1 + args.inscale))
                images = images * scaled_noise[:, None, None, None]

            images = F.dropout(images, p=args.indropout, training=True)
            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            running_xentropy += xentropy_loss.item()

            # Calculate running average of accuracy
            _, hard_pred = torch.max(pred, 1)
            total += labels.size(0)
            correct += (hard_pred == labels).sum().item()
            accuracy = correct / float(total)

            global_step += 1
            progress_bar.set_postfix(xentropy='%.3f' % (running_xentropy / (i + 1)),
                                     acc='%.3f' % accuracy,
                                     lr='%.3e' % cnn_optimizer.param_groups[0]['lr'])

        val_loss, val_acc = evaluate(valid_loader)
        print('Val loss: {:6.4f} | Val acc: {:6.4f}'.format(val_loss, val_acc))
        sys.stdout.flush()
        stats = { 'global_step': global_step, 'time': time.time() - start_time, 'loss': val_loss, 'acc': val_acc }

        if val_loss < stored_loss:
            with open(os.path.join(args.save_dir, 'best_checkpoint.pt'), 'wb') as f:
                torch.save(cnn, f)
            print('Saving model (new best validation)')
            sys.stdout.flush()
            stored_loss = val_loss
            patience_elapsed = 0
        else:
            patience_elapsed += 1

        # Learning rate based on the nonmonotonic criterion
        if len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]):
            print('Decaying the learning rate')
            sys.stdout.flush()
            cnn_optimizer.param_groups[0]['lr'] *= args.lr_decay

        best_val_loss.append(val_loss)

        avg_xentropy = running_xentropy / (i + 1)

        if callback is not None:
            callback(epoch, avg_xentropy, correct / float(total), val_loss, val_acc, config)

        if reporter is not None:
            reporter(timesteps_total=epoch, mean_loss=val_loss)

        if cnn_optimizer.param_groups[0]['lr'] < 1e-7:
            break

        epoch += 1


    # Load best model and run on test
    with open(os.path.join(args.save_dir, 'best_checkpoint.pt'), 'rb') as f:
        cnn = torch.load(f)

    train_loss = avg_xentropy
    train_acc = correct / float(total)

    # Run on val and test data.
    val_loss, val_acc = evaluate(valid_loader)
    test_loss, test_acc = evaluate(test_loader)

    print('=' * 89)
    print('| End of training | trn loss: {:8.5f} | trn acc {:8.5f} | val loss {:8.5f} | val acc {:8.5f} | test loss {:8.5f} | test acc {:8.5f}'.format(
             train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
    print('=' * 89)
    sys.stdout.flush()

    # Save the final val and test performance to a results CSV file
    with open(os.path.join(args.save_dir, 'result_{}.csv'.format(time.time())), 'w') as result_file:
        result_writer = csv.DictWriter(result_file, fieldnames=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
        result_writer.writeheader()
        result_writer.writerow({ 'train_loss': train_loss,
                                 'train_acc': train_acc,
                                 'val_loss': val_loss, 'val_acc': val_acc,
                                 'test_loss': test_loss, 'test_acc': test_acc })
        result_file.flush()

    if return_all:
        print("RETURNING ", train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
        sys.stdout.flush()
        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
    else:
        print("RETURNING ", stored_loss)
        sys.stdout.flush()
        return stored_loss


if __name__ == '__main__':

    config = {
        'model': 'alexnet',

        'dropout0': 0.3,
        'dropout1': 0.3,
        'dropout2': 0.3,
        'dropout3': 0.3,
        'dropout4': 0.3,

        'indropout': 0.4,
        'fc_dropout0': 0.4,
        'fc_dropout1': 0.4,

        'hue': 0.2,
        'saturation': 0.4,
        'brightness': 0.4,
        'contrast': 0.4,

        'cutholes': 2,
        'cutlength': 5,

        'save_dir': 'baseline_saves'
    }

    cnn_val_loss(config=config, reporter=None, callback=None)
