import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, default='test', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='tsp', help='name of problem')

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, nargs='+', default=20,
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=4321, help="Random seed")

    opts = parser.parse_args()

    datadir = os.path.join(opts.data_dir, opts.problem)
    os.makedirs(datadir, exist_ok=True)

    if opts.filename is None:
        filename = os.path.join(datadir, "{}{}_{}_seed{}.pkl".format(
            opts.problem,
            opts.graph_size, opts.name, opts.seed))
    else:
        filename = check_extension(opts.filename)

    np.random.seed(opts.seed)
    dataset = generate_tsp_data(opts.dataset_size, opts.graph_size)
    print(dataset[0])

    save_dataset(dataset, filename)