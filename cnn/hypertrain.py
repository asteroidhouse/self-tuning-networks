import os
import sys
import csv
import ipdb
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

sys.path.insert(0, '..')
sys.path.insert(0, 'hypermodels')

from util.dropout import dropout
from hypermodels.small import SmallCNN
from hypermodels.alexnet import AlexNet
from datasets.cifar import CustomCIFAR10
from datasets.loaders import create_loaders

from util.hyperparameter import create_hparams
from stn_utils.hyperparameter import perturb, hparam_transform, \
                                     hnet_transform, compute_entropy, \
                                     create_hlabels, create_hstats

from logger import Logger


###############################################################################
# Arguments
###############################################################################
model_options = ['small', 'alexnet']

parser = argparse.ArgumentParser(description='CNN')
# Architecture hyperparameters
parser.add_argument('--model', '-m', default='alexnet', choices=model_options)
# Tuning options
parser.add_argument('--tune_scales', '-tscl', action='store_true', default=False, help='whether to tune scales of perturbations by penalizing entropy on training set')
parser.add_argument('--tune_dropout', '-tdrop', action='store_true', default=False, help='whether to tune dropout rate shared across layers')
parser.add_argument('--tune_dropoutl', '-tdropl', action='store_true', default=False, help='whether to tune dropout rates (per layer)')
parser.add_argument('--tune_indropout', '-tidrop', action='store_true', default=False, help='whether to tune dropout rate on input')
parser.add_argument('--tune_fcdropout', '-tfcdrop', action='store_true', default=False, help='whether to tune dropout rate on input')
parser.add_argument('--tune_inscale', '-tiscl', action='store_true', default=False, help='whether to tune scaling applied to input')
parser.add_argument('--tune_hue', '-thue', action='store_true', default=False, help='whether to tune hue jitter')
parser.add_argument('--tune_sat', '-tsat', action='store_true', default=False, help='whether to tune saturation jitter')
parser.add_argument('--tune_contrast', '-tcon', action='store_true', default=False, help='whether to tune contrast jitter')
parser.add_argument('--tune_bright', '-tbrt', action='store_true', default=False, help='whether to tune bright jitter')
parser.add_argument('--tune_jitters', '-tjit', action='store_true', default=False, help='whether to tune hue jitter')
parser.add_argument('--tune_cutlength', '-tcutl', action='store_true', default=False, help='whether to tune length of cutout holes')
parser.add_argument('--tune_cutholes', '-tcuth', action='store_true', default=False, help='whether to tune number of cutout holes')
parser.add_argument('--tune_all', '-tall', action='store_true', default=False, help='whether to tune everything')
# Initial hyperparameter settings
parser.add_argument('--start_inscale', '-iscl', type=float, default=0.05, help='starting input scaling factor')
parser.add_argument('--start_drop', '-drop', type=float, default=0.05, help='starting dropout rate')
parser.add_argument('--start_indrop', '-idrop', type=float, default=0.05, help='starting input dropout rate')
parser.add_argument('--start_fcdrop', '-fcdrop', type=float, default=0.05, help='starting input dropout rate')
parser.add_argument('--start_hue', '-hue', type=float, default=0.05, help='starting hue jitter')
parser.add_argument('--start_sat', '-sat', type=float, default=0.05, help='starting saturation jitter')
parser.add_argument('--start_contrast', '-con', type=float, default=0.05, help='starting contrast jitter')
parser.add_argument('--start_bright', '-brt', type=float, default=0.05, help='starting brightness jitter')
parser.add_argument('--start_cutlength', '-cutl', type=float, default=4., help='starting length of cutout hole') 
parser.add_argument('--start_cutholes', '-cuth', type=float, default=1., help='starting number of holes cutout')
# Optimization hyperparameters
parser.add_argument('--batch_size', '-bsz', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--total_epochs', '-totep', type=int, default=500, help='number of training epochs to run for (warmup epochs are included in the count)')
parser.add_argument('--warmup_epochs', '-wupep', type=int, default=5, help='number of warmup epochs to run for before tuning hyperparameters')
parser.add_argument('--train_lr', '-tlr', type=float, default=0.01, help='learning rate on parameters')
parser.add_argument('--valid_lr', '-vlr', type=float, default=3e-3, help='learning rate on hyperparameters')
parser.add_argument('--scale_lr', '-slr', type=float, default=3e-3, help='learning rate on scales (used if tuning scales)')
parser.add_argument('--momentum', '-mom', type=float, default=0.9, help='amount of momentum on usual parameters')
parser.add_argument('--train_steps', '-tstep', type=int, default=2, help='number of batches to optimize parameters on training set')
parser.add_argument('--valid_steps', '-vstep', type=int, default=1, help='number of batches to optimize hyperparameters on validation set')
# LR decay hyperparameters
parser.add_argument('--lr_decay', type=float, default=0.1, help='Factor by which to multiply the learning rate.')
parser.add_argument('--nonmono', '-nonm', type=int, default=60, help='how many previous epochs to consider for nonmonotonic criterion')
parser.add_argument('--patience', '-pat', type=int, default=75, help='How long to wait for the val loss to improve before early stopping.')
# Regularization hyperparameters
parser.add_argument('--entropy_weight', '-ewt', type=float, default=1e-5, help='penalty applied to entropy of perturbation distribution')
parser.add_argument('--perturb_scale', '-pscl', type=float, default=0.5, help='scale of perturbation applied to continuous hyperparameters')
parser.add_argument('--cutlength_scale', '-clscl', type=float, default=0.5, help='scale of perturbation applied to cutlength hyperparameter')
parser.add_argument('--cutholes_scale', '-chscl', type=float, default=0.5, help='scale of perturbation applied to cutholes hyperparameter')
# Miscellaneous hyperparameters
parser.add_argument('--log_interval', type=int, default=50, help='how many steps before logging stats')
parser.add_argument('--percent_valid', '-pval', type=float, default=0.2, help='percentage of training dataset to be used as a validation set')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save', action='store_true', default=False, help='whether to save current run')
parser.add_argument('--logdir', default='logs', help='directory of regNet to save to')
parser.add_argument('--dir', default='hyper-training', help='directory of logdir folder to save in')
parser.add_argument('--subdir', default='', help='subdirectory of logdir/dir to save in')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.tune_all:
    args.tune_dropoutl = args.tune_indropout = args.tune_inscale =\
        args.tune_jitters = args.tune_cutlength = args.tune_cutholes = True
    args.tune_fcdropout = (args.model == 'alexnet')

assert any([args.tune_dropout, args.tune_dropoutl,
            args.tune_hue, args.tune_sat, args.tune_bright, args.tune_contrast,
            args.tune_jitters, args.tune_indropout, args.tune_cutlength,
            args.tune_cutholes, args.tune_inscale, args.tune_fcdropout]), \
            "Must tune something when hypertraining"


###############################################################################
# Data Loading/Processing
###############################################################################
train_loader, valid_loader, test_loader = create_loaders(args, hyper=True)

train_iter = iter(train_loader)
valid_iter = iter(valid_loader)
###############################################################################
# Model/Optimizer
###############################################################################
num_classes = 10
cnn_class = {'small': SmallCNN, 'alexnet': AlexNet}[args.model]

htensor, hscale, hdict = create_hparams(args, cnn_class, device)

num_hparams = htensor.size(0)

if args.model == 'small':
    cnn = SmallCNN(args, num_classes, num_hparams)
elif args.model == 'alexnet':
    cnn = AlexNet(args, num_classes, num_hparams)

cnn = cnn.to(device)

total_params = sum(param.numel() for param in cnn.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.train_lr, momentum=args.momentum)
hyper_optimizer = torch.optim.Adam([htensor], lr=args.valid_lr)
scale_optimizer = torch.optim.Adam([hscale], lr=args.scale_lr)

###############################################################################
# Saving
###############################################################################
hlabels = create_hlabels(hdict, args)
train_labels = ('global_step', 'train_epoch', 'valid_epoch', 'time', 'loss', 'acc')
valid_labels = ('global_step', 'train_epoch', 'valid_epoch', 'time', 'loss', 'acc', 'entropy')
valid_labels += hlabels
test_labels = ('global_step', 'time', 'loss', 'acc')
epoch_labels = ['epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']

label_dict = { 'train': train_labels, 'valid': valid_labels, 'test': test_labels, 'epoch': epoch_labels }
logger = Logger(sys.argv, args, label_dict)


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([cnn, htensor, hscale, cnn_optimizer, hyper_optimizer], f)

def model_load(fn):
    global cnn, nonfil_htensor, fil_htensor, hscale, cnn_optimizer, hyper_optimizer
    with open(fn, 'rb') as f:
        cnn, htensor, hscale, cnn_optimizer, hyper_optimizer = torch.load(f)

###############################################################################
# Evaluation
###############################################################################
def evaluate(loader):
    """Returns the loss and accuracy on the entire validation/test set.

    Arguments:
    loader -- a DataLoader wrapping around the validation/test set
    """
    cnn.eval()    # Change model to 'eval' mode.
    correct = total = loss = 0.
    loader.dataset.reset_hparams()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            hnet_tensor = hnet_transform(htensor.repeat(images.size(0), 1), hdict)
            hparam_tensor = hparam_transform(htensor.repeat(images.size(0), 1), hdict)
            pred = cnn(images, hnet_tensor, hparam_tensor, hdict)
            loss += F.cross_entropy(pred, labels, reduction='sum').item()
            hard_pred = torch.max(pred, 1)[1]
            total += labels.size(0)
            correct += (hard_pred == labels).sum().item()

    accuracy = correct / total
    mean_loss = loss / total
    return mean_loss, accuracy


###############################################################################
# Optimization step
###############################################################################
def next_batch(data_iter, data_loader, curr_epoch):
    """Load next minibatch."""
    try:
        images, labels = data_iter.next()
    except StopIteration:
        curr_epoch += 1
        data_iter = iter(data_loader)
        images, labels = data_iter.next()

    images, labels = images.to(device), labels.to(device)
    return images, labels, data_iter, curr_epoch


def optimization_step(data_iter, data_loader, curr_epoch, hyper=False):
    cnn_optimizer.zero_grad()
    hyper_optimizer.zero_grad()
    scale_optimizer.zero_grad()
    data_loader.dataset.reset_hparams()

    if not hyper or args.tune_scales:
        batch_htensor = perturb(htensor, hscale, args.batch_size)
    else:
        batch_htensor = htensor.repeat(args.batch_size, 1)

    hparam_tensor = hparam_transform(batch_htensor, hdict)

    if not hyper:
        data_loader.dataset.set_hparams(hparam_tensor, hdict)

    images, labels, data_iter, curr_epoch = next_batch(data_iter, data_loader, curr_epoch)
    hparam_tensor = hparam_tensor[:images.size(0)]
    hnet_tensor = hnet_transform(batch_htensor[:images.size(0)], hdict)

    # Apply input transformations.
    if args.tune_indropout and not hyper:
        indrop_idx = hdict['indropout'].index
        probs = hparam_tensor[:,indrop_idx]
        images = dropout(images, probs, training=True)
    if args.tune_inscale and not hyper:
        inscale_idx = hdict['inscale'].index
        inscale = hparam_tensor[:,inscale_idx]
        noise = torch.rand(images.size(0), device=device)
        scaled_noise = ((1 + inscale) - (1 / (1 + inscale))) * noise + (1/(1 + inscale))
        images = images * scaled_noise[:,None,None,None]

    pred = cnn(images, hnet_tensor, hparam_tensor, hdict)
    xentropy_loss = F.cross_entropy(pred, labels)
    entropy = compute_entropy(hscale)
    loss = xentropy_loss - args.entropy_weight * entropy
    loss.backward()

    if not hyper:
        cnn_optimizer.step()
    else:
        hyper_optimizer.step()
        if args.tune_scales:
            scale_optimizer.step()

    # Calculate number of correct predictions.
    _, hard_pred = torch.max(pred, 1)
    num_correct = (hard_pred == labels).sum().item()
    num_points = labels.size(0)
    step_stats = { 'xentropy': xentropy_loss.item(), 'entropy': entropy.item(),
                   'num_correct': num_correct, 'num_points': num_points }

    return data_iter, curr_epoch, step_stats


###############################################################################
# Training Loop
###############################################################################
# Bookkeeping stuff.
train_step = valid_step = global_step = wup_step = 0
train_epoch = valid_epoch = 0
test_step = 0
start_time = time.time()

train_stats = { 'xentropy': 0., 'num_correct': 0., 'num_points': 0. }
valid_stats = { 'xentropy': 0., 'entropy': 0., 'num_correct': 0., 'num_points': 0. }
epoch_stats = { 'xentropy': 0., 'num_correct': 0., 'num_points': 0. }

def accumulate_stats(running_stats, curr_stats):
    for k in running_stats:
        running_stats[k] += curr_stats[k]

def clear_stats(running_stats):
    for k in running_stats:
        running_stats[k] = 0.

def summarize_stats(running_stats, hyper=False):
    time_taken = time.time() - start_time
    avg_xentropy = running_stats['xentropy'] / args.log_interval
    avg_accuracy = running_stats['num_correct'] / running_stats['num_points']

    summary_stats = { 'global_step': global_step, 'train_epoch': train_epoch,
                      'valid_epoch': valid_epoch, 'time': time_taken, 'loss': avg_xentropy,
                      'acc': avg_accuracy }

    if not hyper:
        return summary_stats
    else:
        avg_entropy = running_stats['entropy'] / args.log_interval
        summary_stats['entropy'] = avg_entropy
        hstats = create_hstats(htensor, hscale, hdict, args)
        summary_stats.update(hstats)
        return summary_stats, hstats

# Warmup for specified number of epochs. Do not tune hyperparameters during this time.
curr_train_epoch = train_epoch

cnn.train()
while train_epoch < args.warmup_epochs:
    train_iter, train_epoch, stats = optimization_step(train_iter, train_loader, train_epoch)
    accumulate_stats(train_stats, stats)
    accumulate_stats(epoch_stats, stats)

    if wup_step % args.log_interval == 0 and global_step > 0:
        summary_stats = summarize_stats(train_stats)
        print('Global Step: {} Train Epoch: {} \tWarmup step:{} \tLoss: {:.3f} \
               Accuracy: {:.3f}'.format(global_step, train_epoch, wup_step, 
                summary_stats['loss'], summary_stats['acc']))
        logger.write('train', summary_stats)
        clear_stats(train_stats)

    wup_step += 1
    global_step += 1

    if curr_train_epoch != train_epoch:
        val_loss, val_acc = evaluate(valid_loader)

        mean_train_loss = epoch_stats['xentropy'] / float(len(train_loader))
        train_acc = epoch_stats['num_correct'] / float(epoch_stats['num_points'])

        elapsed_time = time.time() - start_time

        print('=' * 80)
        print('Train Epoch: {} | Trn Loss: {:.3f} | Trn Acc: {:.3f} | Val Loss: {:.3f} | Val acc: {:.3f}'.format(
               train_epoch, mean_train_loss, train_acc, val_loss, val_acc))
        print('=' * 80)

        epoch_dict = { 'epoch': curr_train_epoch, 'time': elapsed_time,
                       'train_loss': mean_train_loss, 'train_acc': train_acc,
                       'val_loss': val_loss, 'val_acc': val_acc,
                       'lr': cnn_optimizer.param_groups[0]['lr']}
        logger.write('epoch', epoch_dict)

        curr_train_epoch = train_epoch
        clear_stats(epoch_stats)

clear_stats(train_stats)
clear_stats(epoch_stats)

best_val_loss = []
stored_loss = float('inf')
patience_elapsed = 0

scheduler = MultiStepLR(cnn_optimizer, milestones=[60,120,160], gamma=args.lr_decay)

try:
    # Enter main training loop. Alternate between optimizing on training set for
    # args.train_steps and on validation set for args.valid_steps.
    while patience_elapsed < args.patience:
        # Check whether we should use training or validation set.
        cycle_pos = (train_step + valid_step) % (args.train_steps + args.valid_steps)
        hyper = cycle_pos >= args.train_steps

        # Do a step on the training set.
        if not hyper:
            cnn.train()
            curr_train_epoch = train_epoch
            train_iter, train_epoch, stats = (
                optimization_step(train_iter, train_loader,train_epoch))
            changed_epoch = (curr_train_epoch != train_epoch)
            accumulate_stats(train_stats, stats)
            accumulate_stats(epoch_stats, stats)

            if train_step % args.log_interval == 0 and global_step > 0:
                summary_stats = summarize_stats(train_stats)
                print('Global Step: {} Train Epoch: {} \tTrain step:{} \tLoss: {:.3f} Accuracy: {:.3f} lr: {:.4e}'.format(
                       global_step, train_epoch, train_step, summary_stats['loss'], summary_stats['acc'],
                       cnn_optimizer.param_groups[0]['lr']))
                logger.write('train', summary_stats)
                clear_stats(train_stats)

            train_step += 1

        # Do a step on the validation set.
        else:
            cnn.eval()
            valid_iter, valid_epoch, stats = (
                optimization_step(valid_iter, valid_loader, valid_epoch, hyper=True))
            accumulate_stats(valid_stats, stats)

            if valid_step % args.log_interval == 0 and global_step > 0:
                summary_stats, hstats = summarize_stats(valid_stats, hyper=True)
                print('Global Step: {} Valid Epoch: {} \t Valid Step {} \
                    \tLoss: {:.6f} Accuracy: {:.3f} Entropy {:.3f}'.format(global_step,
                        valid_epoch, valid_step, summary_stats['loss'],
                        summary_stats['acc'], summary_stats['entropy']))
                logger.write('valid', summary_stats)
                clear_stats(valid_stats)

            valid_step += 1

        global_step += 1

        # If just completed an epoch on the training set, check the test loss.
        if changed_epoch:

            changed_epoch = False  # Reset changed_epoch back to False

            val_loss, val_acc = evaluate(valid_loader)

            mean_train_loss = epoch_stats['xentropy'] / float(len(train_loader))
            train_acc = epoch_stats['num_correct'] / float(epoch_stats['num_points'])

            elapsed_time = time.time() - start_time

            print('=' * 80)
            print('Train Epoch: {} | Trn Loss: {:.3f} | Trn Acc: {:.3f} | Val Loss: {:.3f} | Val acc: {:.3f}'.format(
                   curr_train_epoch, mean_train_loss, train_acc, val_loss, val_acc))
            print('=' * 80)

            epoch_dict = { 'epoch': curr_train_epoch, 'time': elapsed_time,
                           'train_loss': mean_train_loss, 'train_acc': train_acc,
                           'val_loss': val_loss, 'val_acc': val_acc,
                           'lr': cnn_optimizer.param_groups[0]['lr']}
            logger.write('epoch', epoch_dict)

            if (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                
                cnn_optimizer.param_groups[0]['lr'] *= args.lr_decay
                print('Decaying the learning rate to {}'.format(
                    cnn_optimizer.param_groups[0]['lr']))
                sys.stdout.flush()

            best_val_loss.append(val_loss)

            if val_loss < stored_loss:
                model_save(os.path.join(logger.logdir, 'best_checkpoint.pt'))
                print('Saving model (new best validation)')
                sys.stdout.flush()
                stored_loss = val_loss
                patience_elapsed = 0
            else:
                patience_elapsed += 1

            if cnn_optimizer.param_groups[0]['lr'] < 1e-5:  # Another stopping criterion based on decaying the lr
                break

            clear_stats(epoch_stats)

except KeyboardInterrupt:
    print('=' * 89)
    print('Exiting from training early')
    sys.stdout.flush()


# Load the best saved model.
model_load(os.path.join(logger.logdir, 'best_checkpoint.pt'))

# Run on val and test data.
val_loss, val_acc = evaluate(valid_loader)
test_loss, test_acc = evaluate(test_loader)

print('=' * 89)
print('| End of training | val loss {:8.5f} | val acc {:8.5f} | test loss {:8.5f} | test acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc))
print('=' * 89)
sys.stdout.flush()

# Save the final val and test performance to a results CSV file
with open(os.path.join(logger.logdir, 'result.csv'), 'w') as result_file:
    result_writer = csv.DictWriter(result_file,
        fieldnames=['val_loss', 'val_acc', 'test_loss', 'test_acc'])
    result_writer.writeheader()
    result_writer.writerow({ 'val_loss': val_loss, 'val_acc': val_acc,
        'test_loss': test_loss, 'test_acc': test_acc })
    result_file.flush()

if args.save:
    logger.close()
