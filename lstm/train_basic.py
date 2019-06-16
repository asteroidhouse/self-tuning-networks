import os
import sys
import csv
import ipdb
import time
import math
import hashlib
import datetime
import argparse

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# Local imports
import data
import model_basic as model

from logger import Logger
from utils import batchify, get_batch, repackage_hidden


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')

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
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='Number of epochs for nonmonotonic lr decay')
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Flag to DISABLE CUDA (ENABLED by default)')
parser.add_argument('--gpu', type=int, default=0,
                    help='Select which GPU to use (e.g., 0, 1, 2, or 3)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0,
                    help='weight decay applied to all weights')

parser.add_argument('--lr_decay', type=float, default=4.0,
                    help='Learning rate decay.')
parser.add_argument('--patience', type=int, default=20,
                    help='How long to wait for the val loss to improve before early stopping.')

parser.add_argument('--perturb_dropoute', action='store_true', default=False,
                    help='Perturb embedding dropout (with Gaussian or sinusoid noise)')
parser.add_argument('--perturb_dropouti', action='store_true', default=False,
                    help='Perturb input dropout (with Gaussian or sinusoid noise)')
parser.add_argument('--perturb_dropouth', action='store_true', default=False,
                    help='Perturb hidden dropout (with Gaussian or sinusoid noise)')
parser.add_argument('--perturb_dropouto', action='store_true', default=False,
                    help='Perturb output dropout (with Gaussian or sinusoid noise)')
parser.add_argument('--perturb_type', type=str, default='gaussian', choices=['gaussian', 'sinusoid'],
                    help='Choose the method to perturb the hyperparameters')
parser.add_argument('--perturb_std', type=float, default=0.1,
                    help='The standard deviation of the gaussian hyperparameter perturbations.')
parser.add_argument('--amplitude', type=float, default=0.2,
                    help='The amplitude of sinusoid perturbations on hyperparameter values.')
parser.add_argument('--sinusoid_period', type=float, default=330,
                    help='The period of sinusoid perturbations, measured in mini-batches.')

parser.add_argument('--save_dir', type=str, default='saves',
                    help='The base save directory.')
parser.add_argument('--prefix', type=str, default=None,
                    help='An optional prefix for the experiment name -- for uniqueness.')
parser.add_argument('--test', action='store_true', default=False,
                    help="Just run test, don't train.")
parser.add_argument('--overwrite', action='store_true', default=False,
                    help="Run the experiment and overwrite a (possibly existing) result file.")
parser.add_argument('--load_schedule', type=str, default=None,
                    help='Optionally, load a csv file that defines a per-epoch or per-iteration hyperparameter schedule.')

args = parser.parse_args()
args.tied = True

if not args.disable_cuda and torch.cuda.is_available():
    use_device = torch.device('cuda:{}'.format(args.gpu))
else:
    use_device = torch.device('cpu')

# Set the random seed manually for reproducibility.
if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


# Create hyperparameters and logger
# ---------------------------------
files_used = ['train_basic.py', 'utils.py', 'model_basic.py']
epoch_labels = ['epoch', 'time', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl']
stats = { 'epoch': epoch_labels }
logger = Logger(sys.argv, args, files=files_used, stats=stats)
# ---------------------------------


# Initialize sinusoid hyperparameter adaptation
args.multiplier_increment = (2 * np.pi) / args.sinusoid_period
# Start the multiplier for each hyperparameter at a different initial value
multipliers = {hparam: np.random.rand() * args.sinusoid_period for hparam in ['dropoute', 'dropouti', 'dropouth', 'dropouto']}

# Base saves folder
BASE_SAVE_DIR = logger.log_dir


###############################################################################
# Create save folder
###############################################################################

if not args.test:
    # Load schedule, if provided
    if args.load_schedule:
        hyper_schedule = {}

        with open(args.load_schedule, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                hyper_schedule[int(row['epoch'])] = { k: float(v) for (k,v) in row.items() if k not in ['epoch', 'time', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl'] }

    save_dir = logger.log_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if the result file exists, and if so, don't run it again.
    if not args.overwrite:
        if os.path.exists(os.path.join(save_dir, 'result')):
            print("The result file {} exists! Not rerunning.".format(os.path.join(save_dir, 'result')))
            sys.exit(0)

    with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, use_device)
val_data = batchify(corpus.valid, eval_batch_size, use_device)
test_data = batchify(corpus.test, test_batch_size, use_device)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropouto, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
model = model.to(use_device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(use_device)
###
params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).data
            hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


global_iteration = 0

def train():

    global global_iteration

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0

    train_losses = []

    while i < train_data.size(0) - 1 - 1:

        seq_len = args.bptt

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # Optionally perturb hyperparameter values (via Gaussian or sinusoid perturbations)
        # ---------------------------------------------------------------------------------
        for hparam in ['dropoute', 'dropouti', 'dropouth', 'dropouto']:
            if getattr(args, 'perturb_'+hparam):
                if args.perturb_type == 'gaussian':
                    gaussian_perturbation = args.perturb_std * np.random.randn()
                    use_hparam_value = getattr(args, hparam) + gaussian_perturbation
                elif args.perturb_type == 'sinusoid':
                    use_hparam_value = getattr(args, hparam) + args.amplitude * np.sin(multipliers[hparam] * 2 * np.pi)
                    multipliers[hparam] += args.multiplier_increment

                use_hparam_value = np.clip(use_hparam_value, 0.0, 0.99)
                setattr(model, hparam, use_hparam_value)
        # ---------------------------------------------------------------------------------

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        loss = raw_loss
        # Activation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        train_losses.append(raw_loss.item())

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time

            val_loss = evaluate(val_data, eval_batch_size)

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | trn ppl {:8.2f} | val ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), math.exp(val_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        ###
        batch += 1
        i += seq_len
        global_iteration += 1

    return np.mean(train_losses)

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
start_time = time.process_time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)

    epoch = 0
    patience_elapsed = 0

    while epoch < args.epochs and patience_elapsed < args.patience:

        if args.load_schedule:
            if epoch in hyper_schedule:
                epoch_dict = hyper_schedule[epoch]
                for hparam in epoch_dict:
                    setattr(model, hparam, epoch_dict[hparam])
                    print("Set hyperparameter {} = {}".format(hparam, epoch_dict[hparam]))

        epoch_start_time = time.time()
        train_loss = train()

        val_loss = evaluate(val_data, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)

        # TRAINING LOG
        elapsed_time = time.process_time() - start_time
        epoch_stats = { 'epoch': epoch, 'time': elapsed_time, 'train_loss': train_loss, 'val_loss': val_loss,
                        'train_ppl': math.exp(train_loss), 'val_ppl': math.exp(val_loss) }
        logger.write('epoch', epoch_stats)

        if val_loss < stored_loss:
            model_save(os.path.join(save_dir, args.save))
            print('Saving model (new best validation)')
            sys.stdout.flush()
            stored_loss = val_loss
            patience_elapsed = 0
        else:
            patience_elapsed += 1

        if len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]):
            print('Decaying the learning rate')
            optimizer.param_groups[0]['lr'] /= args.lr_decay

        if optimizer.param_groups[0]['lr'] < 0.0003:  # Early stopping criterion based on decaying the lr
            break

        best_val_loss.append(val_loss)
        epoch += 1
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(save_dir, args.save))

# Run on val and test data.
val_loss = evaluate(val_data, eval_batch_size)
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | val loss {:8.5f} | val ppl {:8.5f} | test loss {:8.5f} | test ppl {:8.5f}'.format(
       val_loss, math.exp(val_loss), test_loss, math.exp(test_loss)))
print('=' * 89)
sys.stdout.flush()

# Save the final val and test performance to a results file
with open(os.path.join(logger.log_dir, 'result.csv'), 'w') as result_file:
    result_writer = csv.DictWriter(result_file, fieldnames=['val_loss', 'val_ppl', 'test_loss', 'test_ppl'])
    result_writer.writeheader()
    result_writer.writerow({ 'val_loss': val_loss, 'val_ppl': math.exp(val_loss), 'test_loss': test_loss, 'test_ppl': math.exp(test_loss) })
    result_file.flush()
