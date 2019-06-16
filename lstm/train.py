"""Training a Self-Tuning LSTM.
"""
import os
import sys
import csv
import ipdb
import time
import math
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Local imports
import data
import hyperlstm

import utils
from utils import batchify, get_batch, repackage_hidden

from logger import Logger

sys.path.insert(0, '..')
from stn_utils.hyperparameter import perturb, hparam_transform, hnet_transform, compute_entropy, \
                                     create_hlabels, create_hstats


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--hyper_lr', type=float, default=0.01, metavar='LR',
                    help='Learning rate for hyperparameter optimizer.')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')

parser.add_argument('--train_steps', type=int, default=2,
                    help='number of batches to optimize parameters on training set')
parser.add_argument('--val_steps', type=int, default=1,
                    help='number of batches to optimize hyperparameters on validation set')
parser.add_argument('--warmup_steps', type=int, default=331,  # The number of minibatches in one epoch
                    help='number of batches to optimize parameters on training set before starting to alternate')
parser.add_argument('--perturb_scale', type=float, default=1.0,
                    help='Variance of normal distribution for sampling logit perturbations.')

######################################
### Regularization hyperparameters ###
######################################
parser.add_argument('--dropouto', type=float, default=0.05,
                    help='dropout applied to output (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.05,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.05,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.05,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.05,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0,
                    help='weight decay applied to all weights')

parser.add_argument('--drop_transform', default='none', choices=['none', 'sigmoid'],
                    help='what transformation is applied to dropouts before being fed into hypernet')
parser.add_argument('--alpha_beta_transform', default='none', choices=['none', 'softplus'],
                    help='what transformation is applied to alpha and beta before being fed into hypernet')

parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Flag to DISABLE CUDA (ENABLED by default)')
parser.add_argument('--gpu', type=int, default=0,
                    help='Select which GPU to use (e.g., 0, 1, 2, or 3)')

parser.add_argument('--train_log_interval', type=int, default=1, metavar='N',
                    help='Training loss/stats report interval')
parser.add_argument('--val_log_interval', type=int, default=1, metavar='N',
                    help='Validation loss/stats report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')

parser.add_argument('--prefix', type=str, default=None,
                    help='An optional prefix for the experiment name -- for uniqueness.')
parser.add_argument('--test', action='store_true', default=False,
                    help="Just run test, don't train.")

# Tune dropout hyperparameters
parser.add_argument('--tune_wdrop', action='store_true', default=False,
                    help='Tune the weight dropconnect.')
parser.add_argument('--tune_dropoute', action='store_true', default=False,
                    help='Tune the embedding dropout.')
parser.add_argument('--tune_dropouti', action='store_true', default=False,
                    help='Tune the input dropout.')
parser.add_argument('--tune_dropouto', action='store_true', default=False,
                    help='Tune the output dropout.')
parser.add_argument('--tune_dropouth', action='store_true', default=False,
                    help='Tune the dropout between layers.')
parser.add_argument('--tune_alpha', action='store_true', default=False,
                    help='Tune the coefficient for activation regularization.')
parser.add_argument('--tune_beta', action='store_true', default=False,
                    help='Tune the coefficient for temporal activation regularization.')
parser.add_argument('--tune_all', action='store_true', default=False,
                    help='Tune all regularization hyperparameters.')

parser.add_argument('--save_dir', type=str, default='saves',
                    help='The base save directory.')
parser.add_argument('--lr_decay', type=float, default=4.0,
                    help='Learning rate decay.')
parser.add_argument('--patience', type=int, default=20,
                    help='How long to wait for the val loss to improve before early stopping.')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help="Run the experiment and overwrite a (possibly existing) result file.")

args = parser.parse_args()
args.tied = True

if args.tune_all:
    args.tune_wdrop = args.tune_dropoute = args.tune_dropouti = args.tune_dropouth =\
        args.tune_dropouto = args.tune_alpha = args.tune_beta = True
    args.drop_transform = 'sigmoid'
    args.alpha_beta_transform = 'softplus'
    # Results are more robust if you replace sigmoid with unconstrained parameterization (below)
    # args.drop_transform = 'none'
    # args.alpha_beta_transform = 'none'
    args.alpha = args.beta = 0.05

args.hypertrain = (args.tune_wdrop or args.tune_dropoute or args.tune_dropouti or args.tune_dropouth or args.tune_dropouto or \
                   args.tune_alpha or args.tune_beta)


if not args.disable_cuda and torch.cuda.is_available():
    use_device = torch.device('cuda:{}'.format(args.gpu))
else:
    use_device = torch.device('cpu')

if args.seed:
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


# Create hyperparameters and logger
# ---------------------------------
files_used = ['train.py', 'utils.py', 'hyperlstm.py']

htensor, hscale, hdict = utils.create_hparams(args, use_device)
num_hparams = htensor.size(0)
hlabels = create_hlabels(hdict, args)

val_labels = ['global_step', 'time']
val_labels += hlabels
epoch_labels = ['epoch', 'time', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl']
epoch_labels += hlabels

stats = { 'val': val_labels, 'epoch': epoch_labels }
logger = Logger(sys.argv, args, files=files_used, stats=stats)


###############################################################################
# Load data
###############################################################################
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, htensor, hscale, criterion, param_optimizer, hparam_optimizer], f)


def model_load(fn):
    global model, htensor, hscale, hdict, criterion, param_optimizer, hparam_optimizer
    with open(fn, 'rb') as f:
        model, htensor, hscale, criterion, param_optimizer, hparam_optimizer = torch.load(f)


corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, use_device)
hyperval_data = batchify(corpus.valid, args.batch_size, use_device)
val_data = batchify(corpus.valid, eval_batch_size, use_device)
test_data = batchify(corpus.test, test_batch_size, use_device)
ntokens = len(corpus.dictionary)


###############################################################################
# Build the model
###############################################################################
model = hyperlstm.HyperLSTM(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nlayers=args.nlayers, dropouto=args.dropouto, dropouth=args.dropouth,
                            dropouti=args.dropouti, dropoute=args.dropoute, wdrop=args.wdrop, tie_weights=args.tied, num_hparams=num_hparams)

model = model.to(use_device)
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)

            hnet_tensor = hnet_transform(htensor.repeat(data.size(1), 1), hdict)
            hparam_tensor = hparam_transform(htensor.repeat(data.size(1), 1), hdict)

            output, hidden = model(data, hidden, hnet_tensor, hparam_tensor, hdict)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)


val_hidden = model.init_hidden(args.batch_size)

def train_hyperparams(data_source, val_epoch, val_iter, val_seq_pos):

    global global_step, val_hidden

    model.eval()  # Turn on evaluation mode which disables dropout.

    total_loss = 0
    cur_start_time = time.time()
    ntokens = len(corpus.dictionary)
    val_steps_taken = 0

    num_val_batches = len(data_source) // args.bptt

    for batch_idx in range(args.val_steps):

        seq_len = args.bptt

        if val_seq_pos >= len(data_source):
            val_epoch += 1
            val_seq_pos = 0
            val_hidden = model.init_hidden(args.batch_size)

        data, targets = get_batch(data_source, val_seq_pos, args, seq_len=seq_len)

        batch_htensor = htensor.repeat(args.batch_size, 1)
        hparam_tensor = hparam_transform(batch_htensor, hdict)
        hnet_tensor = hnet_transform(batch_htensor, hdict)

        hparam_optimizer.zero_grad()
        val_hidden = repackage_hidden(val_hidden)

        output, val_hidden, rnn_hs, dropped_rnn_hs = model(data, val_hidden, hnet_tensor, hparam_tensor, hdict, return_h=True)

        xentropy_loss = criterion(output.view(-1, ntokens), targets)

        # Activation Regularization
        # ================================================================
        if 'alpha' in hdict:
            alpha = hparam_tensor[:, hdict['alpha'].index][0].item()
        else:
            alpha = args.alpha

        if 'beta' in hdict:
            beta = hparam_tensor[:, hdict['beta'].index][0].item()
        else:
            beta = args.beta

        loss = xentropy_loss
        loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        # ================================================================

        loss.backward()
        hparam_optimizer.step()

        total_loss += loss.item()
        val_steps_taken += 1

        if val_iter % args.val_log_interval == 0:
            cur_loss = total_loss / val_steps_taken
            cur_elapsed = time.time() - cur_start_time
            total_elapsed = time.process_time() - start_time

            # Log hyperparameters
            hstats = create_hstats(htensor, hscale, hdict, args)
            val_stats = { 'global_step': global_step, 'time': total_elapsed }
            val_stats.update(hstats)
            logger.write('val', val_stats)

            # Form a string for printing hparam values
            hparam_string = ' | '.join(["{}: {}".format(hname, hstats[hname]) for hname in hstats])
            print('| Val epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | {}'.format(
                val_epoch, val_iter, num_val_batches, hparam_optimizer.param_groups[0]['lr'],
                cur_elapsed * 1000 / args.val_log_interval, cur_loss, math.exp(cur_loss), hparam_string))
            sys.stdout.flush()

            total_loss = 0
            val_steps_taken = 0
            cur_start_time = time.time()

        val_iter += 1
        val_seq_pos += seq_len
        global_step += 1

    if val_seq_pos >= len(data_source):
        val_epoch += 1
        val_seq_pos = 0
        val_hidden = model.init_hidden(args.batch_size)

    return val_epoch, val_iter, val_seq_pos


global_step = 0
epoch_train_loss = 0
train_hidden = model.init_hidden(args.batch_size)

def train_params(train_epoch, train_iter, train_seq_pos, num_steps):
    global epoch_train_loss, global_step, train_hidden

    total_train_loss = 0
    cur_start_time = time.time()
    ntokens = len(corpus.dictionary)
    train_steps_taken = 0

    num_batches = len(train_data) // args.bptt

    total_train_loss = 0
    train_losses = []

    for batch_idx in range(num_steps):

        seq_len = args.bptt

        if train_seq_pos >= len(train_data):
            train_epoch += 1
            train_seq_pos = 0
            train_hidden = model.init_hidden(args.batch_size)

        lr2 = param_optimizer.param_groups[0]['lr']
        param_optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        # Turn on training mode which enables dropout.
        model.train()
        data, targets = get_batch(train_data, train_seq_pos, args, seq_len=seq_len)

        # Perturb hyperparameters so that the model can learn a local response.
        perturb_htensor = perturb(htensor, hscale, data.size(1), hdict)
        hparam_tensor = hparam_transform(perturb_htensor, hdict)
        hnet_tensor = hnet_transform(perturb_htensor, hdict)
        train_hidden = repackage_hidden(train_hidden)

        output, train_hidden, rnn_hs, dropped_rnn_hs = model(data, train_hidden, hnet_tensor, hparam_tensor, hdict, return_h=True)

        raw_loss = criterion(output.view(-1, ntokens), targets)
        loss = raw_loss

        # Activation Regularization
        # ================================================================
        if 'alpha' in hdict:
            alpha = hparam_tensor[:, hdict['alpha'].index][0].item()
        else:
            alpha = args.alpha

        if 'beta' in hdict:
            beta = hparam_tensor[:, hdict['beta'].index][0].item()
        else:
            beta = args.beta

        loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        # ================================================================

        param_optimizer.zero_grad()
        loss.backward()

        if args.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        param_optimizer.step()

        train_losses.append(loss.item())
        total_train_loss += loss.item()
        epoch_train_loss += loss.item()
        train_steps_taken += 1
        param_optimizer.param_groups[0]['lr'] = lr2

        if train_iter % args.train_log_interval == 0:
            cur_loss = total_train_loss / train_steps_taken
            cur_elapsed = time.time() - cur_start_time
            total_elapsed = time.process_time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                     train_epoch, train_iter, num_batches, param_optimizer.param_groups[0]['lr'],
                     cur_elapsed * 1000 / args.train_log_interval, cur_loss, math.exp(cur_loss)))
            sys.stdout.flush()

            total_train_loss = 0
            train_steps_taken = 0
            cur_start_time = time.time()

        train_iter += 1
        train_seq_pos += seq_len
        global_step += 1

    if train_seq_pos >= len(train_data):
        train_epoch += 1
        train_seq_pos = 0
        train_hidden = model.init_hidden(args.batch_size)

    return np.mean(train_losses), train_epoch, train_iter, train_seq_pos


lr = args.lr
best_val_loss = []
stored_loss = 100000000

start_time = time.process_time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    param_optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    hparam_optimizer = torch.optim.Adam([htensor], lr=args.hyper_lr)

    train_epoch, val_epoch = 0, 0
    train_iter, val_iter = 0, 0
    train_seq_pos, val_seq_pos = 0, 0

    patience_elapsed = 0

    if args.warmup_steps > 0:
        model.train()
        train_loss, train_epoch, train_iter, train_seq_pos = train_params(train_epoch, train_iter, train_seq_pos, args.warmup_steps)

    old_train_epoch = 0

    timed_performance_list = []  # To store tuples of (total_time_elapsed, val_loss)

    while train_epoch < args.epochs and patience_elapsed < args.patience:

        epoch_start_time = time.time()

        model.train()
        train_loss, train_epoch, train_iter, train_seq_pos = train_params(train_epoch, train_iter, train_seq_pos, args.train_steps)

        if args.hypertrain:
            model.eval()
            val_epoch, val_iter, val_seq_pos = train_hyperparams(hyperval_data, val_epoch, val_iter, val_seq_pos)

        # Once we have completed a whole epoch on the training set
        if train_epoch != old_train_epoch:
            num_train_batches = len(train_data) // args.bptt
            mean_epoch_train_loss = epoch_train_loss / num_train_batches

            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | lr: {:6.4f} | train loss {:5.2f} | train ppl {:8.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(train_epoch, (time.time() - epoch_start_time), param_optimizer.param_groups[0]['lr'],
                                               mean_epoch_train_loss, math.exp(mean_epoch_train_loss), val_loss, math.exp(val_loss)))
            print('-' * 89)
            sys.stdout.flush()

            ###########################
            ###    TRAINING LOG     ###
            ###########################
            elapsed_time = time.process_time() - start_time
            timed_performance_list.append((elapsed_time, math.exp(val_loss)))

            epoch_stats = { 'epoch': old_train_epoch, 'time': elapsed_time, 'train_loss': mean_epoch_train_loss, 'val_loss': val_loss,
                            'train_ppl': math.exp(mean_epoch_train_loss), 'val_ppl': math.exp(val_loss) }
            hstats = create_hstats(htensor, hscale, hdict, args)
            epoch_stats.update(hstats)
            logger.write('epoch', epoch_stats)

            if len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]):
                print('Decaying the learning rate')
                sys.stdout.flush()
                param_optimizer.param_groups[0]['lr'] /= args.lr_decay

            old_train_epoch = train_epoch
            epoch_train_loss = 0

            if val_loss < stored_loss:
                model_save(os.path.join(logger.log_dir, args.save))
                print('Saving model (new best validation)')
                sys.stdout.flush()
                stored_loss = val_loss
                patience_elapsed = 0
            else:
                patience_elapsed += 1

            if param_optimizer.param_groups[0]['lr'] < 0.0003:  # A stopping criterion based on decaying the lr
                break

            best_val_loss.append(val_loss)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.stdout.flush()


# Load the best saved model.
model_load(os.path.join(logger.log_dir, args.save))

# Run on val and test data.
val_loss = evaluate(val_data, eval_batch_size)
test_loss = evaluate(test_data, test_batch_size)

print('=' * 89)
print('| End of training | val loss {:8.5f} | val ppl {:8.5f} | test loss {:8.5f} | test ppl {:8.5f}'.format(
       val_loss, math.exp(val_loss), test_loss, math.exp(test_loss)))
print('=' * 89)
sys.stdout.flush()

# Save the final val and test performance to a results CSV file
with open(os.path.join(logger.log_dir, 'result.csv'), 'w') as result_file:
    result_writer = csv.DictWriter(result_file, fieldnames=['val_loss', 'val_ppl', 'test_loss', 'test_ppl'])
    result_writer.writeheader()
    result_writer.writerow({ 'val_loss': val_loss, 'val_ppl': math.exp(val_loss), 'test_loss': test_loss, 'test_ppl': math.exp(test_loss) })
    result_file.flush()
