import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils.functions import move_to, load_model
from torch.utils.data import DataLoader
from utils.data_utils import save_dataset
import time
from datetime import timedelta



def get_best(sequences, cost, ids=None, batch_size=None):
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]
    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0]+1])
    mincosts = np.minimum.reduceat(cost, splits)

    groups_len = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, groups_len) == cost)
    result = np.full(len(groups_len) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]



def _eval_dataset(model, dataset, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type('greedy')

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()

        with torch.no_grad():
            batch_rep = 1
            iter_rep = 1
            sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            batch_size = len(costs)
            ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )

        duration = time.time() - start

        for seq, costs in zip(sequences, costs):
            seq = seq.tolist()
            results.append((costs, seq, duration))

    return results

def eval_dataset(dataset_path, opts):
    model = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
    results = _eval_dataset(model, dataset, opts, device)

    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")

    parser.add_argument('--decode_strategy', type=str, default='greedy',
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')

    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    for dataset_path in opts.datasets:
        eval_dataset(dataset_path, opts)
