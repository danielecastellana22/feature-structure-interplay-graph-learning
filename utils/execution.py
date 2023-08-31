import os
import os.path as osp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .serialisation import from_json_file
from .configuration import Config
from .misc import eprint


def parallel_model_selection(train_config_fun,
                             exp_config: Config,
                             n_splits: int,
                             base_dir: str,
                             resume: bool,
                             max_num_process: int):
    # save the grid search
    exp_config.to_yaml_file(os.path.join(base_dir, 'grid.yaml'))
    config_list = exp_config.build_config_grid()
    n_configs = len(config_list)

    if max_num_process > 1 and n_configs > 1:
        process_pool = ProcessPoolExecutor(max_num_process)
    else:
        process_pool = None
        eprint('No process pool created! The execution will be sequential!')

    print(f'Model selection with {len(config_list)} configurations started!')

    for config_idx, config in enumerate(config_list):
        for split_idx in range(n_splits):
            exp_dir = os.path.join(base_dir, f'config_{config_idx}', f'split_{split_idx}')
            if resume and os.path.exists(osp.join(exp_dir,'training_results.json')):
                eprint( f'config_{config_idx} split_{split_idx} is already done!')

            os.makedirs(exp_dir, exist_ok=resume)
            config.to_yaml_file(os.path.join(exp_dir, 'config.yml'))

            params = {'config': config, 'split_idx': split_idx, 'exp_dir': exp_dir,
                      'write_on_console': process_pool is None}

            if process_pool is None:
                train_config_fun(**params)
            else:
                f = process_pool.submit(train_config_fun, **params)

    if process_pool is not None:
        process_pool.shutdown()

    print(f'Model selection terminated!')


def check_results(result_dir_path, do_print=True):
    count_config_finished = 0

    exp_config = Config.from_yaml_file(os.path.join(result_dir_path, 'grid.yaml'))

    # compute the number of config dir
    list_config_dir_name = [x for x in os.listdir(result_dir_path) if x.startswith('config')]
    n_configs = len(list_config_dir_name)

    # compute the number of split dir
    list_split_dir_name = [x for x in os.listdir(osp.join(result_dir_path,'config_0')) if x.startswith('split')]
    n_splits = len(list_split_dir_name)

    # compute the number of layer dir (if any)
    list_layer_dir_name = [x for x in os.listdir(osp.join(result_dir_path, 'config_0', 'split_0'))
                           if x.startswith('layer')]
    layerwise_results = len(list_layer_dir_name) == 0
    if layerwise_results:
        list_layer_dir_name = ['']

    n_layers = len(list_layer_dir_name)

    all_results = np.empty((n_configs, n_layers, n_splits), dtype=object)
    tot_results = n_configs * n_layers * n_splits

    dict_keys = None
    for config_dir_name in list_config_dir_name:
        config_idx = int(config_dir_name.split('_')[-1])
        for split_dir_name in list_split_dir_name:
            split_idx = int(split_dir_name.split('_')[-1])
            for layer_dir_name in list_layer_dir_name:
                split_dir_path = osp.join(result_dir_path, config_dir_name, split_dir_name, layer_dir_name)
                if layerwise_results:
                    layer_idx = 0
                else:
                    layer_idx = int(layer_dir_name.split('_')[-1])

                results_file = osp.join(split_dir_path, 'training_results.json')
                if osp.exists(results_file):
                    count_config_finished += 1
                    d = from_json_file(results_file)
                    dict_keys = d.keys()
                    all_results[config_idx, layer_idx, split_idx] = d

    numpy_d = {}
    if count_config_finished > 0:

        for k in dict_keys:
            numpy_d[k] = np.zeros_like(all_results, dtype=float)
            for i in range(n_configs):
                for j in range(n_layers):
                    for t in range(n_splits):
                        if all_results[i][j][t] is not None:
                            numpy_d[k][i,j,t] = all_results[i,j,t][k]

            numpy_d[k] = numpy_d[k].squeeze()

        all_train_metric = numpy_d['training_metric'].reshape(-1, n_splits)
        all_val_metric = numpy_d['validation_metric'].reshape(-1, n_splits)
        all_test_metric = numpy_d['test_metric'].reshape(-1, n_splits)

        if do_print:
            print('-'*100)
            print('METHOD 1')
            best_config_idx = np.argmax(np.mean(all_val_metric, axis=-1))
            print(f'{count_config_finished}/{tot_results} finished!')
            print(f'Best config id {best_config_idx}')
            print(f'Best training metric: {np.mean(all_train_metric[best_config_idx, :], axis=-1)*100:0.2f} +/- {np.std(all_train_metric[best_config_idx, :], axis=-1)*100:0.2f}')
            print(f'Best validation metric: {np.mean(all_val_metric[best_config_idx, :], axis=-1)*100:0.2f} +/- {np.std(all_val_metric[best_config_idx, :], axis=-1)*100:0.2f}')
            print(f'Best test metric: {np.mean(all_test_metric[best_config_idx, :], axis=-1)*100:0.2f} +/- {np.std(all_test_metric[best_config_idx, :], axis=-1)*100:0.2f}')

            print('-'*100)
            print('METHOD 2')
            best_config_idx = np.argmax(all_val_metric, axis=0)
            print(f'{count_config_finished}/{tot_results} finished!')
            print(f'Best config ids {best_config_idx}')
            print(f'Best training metric: {np.mean(all_train_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f} +/- {np.std(all_train_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f}')
            print(f'Best validation metric: {np.mean(all_val_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f} +/- {np.std(all_val_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f}')
            print(f'Best test metric: {np.mean(all_test_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f} +/- {np.std(all_test_metric[best_config_idx, np.arange(n_splits)], axis=-1)*100:0.2f}')

    return numpy_d, exp_config