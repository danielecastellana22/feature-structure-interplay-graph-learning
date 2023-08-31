import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch as th
import argparse
from datasets.utils import get_dataset
from utils.execution import parallel_model_selection
from utils.configuration import Config
from utils.misc import create_datatime_dir, eprint, string2class


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--results-dir', dest='results_dir', default='results')
    parser.add_argument('--data-dir', dest='data_dir', default='data')
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--resume-dir', dest='resume_dir', default=None)
    parser.add_argument('--num-workers', dest='num_workers', default=10, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        th.set_anomaly_enabled(True)

    if args.resume_dir is not None:
        eprint('Resuming experiments! results-dir and config-file args will be ignored!')
        base_dir = args.resume_dir
        exp_config = Config.from_yaml_file(os.path.join(base_dir,'grid.yaml'))
    else:
        # read the config dict
        exp_config = Config.from_yaml_file(args.config_file)
        dataset_config = exp_config.dataset_config
        # create base directory for the experiment
        dir_name = os.path.splitext(os.path.basename(args.config_file))[0]
        base_dir = os.path.join(args.results_dir, dir_name)
        base_dir = create_datatime_dir(base_dir)

    # load the dataset just to start the download if needed
    ds = get_dataset(args.data_dir, dataset_config)
    exp_config['storage_dir'] = args.data_dir

    # select the training function according to the model class
    train_fun = string2class(exp_config.model_config['class']).get_training_fun()

    parallel_model_selection(
        train_config_fun=train_fun,
        exp_config=exp_config,
        n_splits=ds.n_splits,
        base_dir=base_dir,
        resume=args.resume_dir is not None,
        max_num_process=args.num_workers if not args.debug else 1)
