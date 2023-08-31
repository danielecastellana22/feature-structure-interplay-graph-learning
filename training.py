import os
import torch as th
from utils.serialisation import to_json_file, to_torch_file
from utils.configuration import create_object_from_config
from utils.misc import get_logger
from datasets.utils import get_dataset
import time


def __train_loop__(dataset, model, optimiser, split_idx, n_epochs, patience, txt_logger, **other_model_forward_params):

    x, edge_index, y = dataset.x, dataset.edge_index, dataset.y
    train_mask, val_mask, test_mask = dataset.train_mask[:, split_idx], dataset.val_mask[:, split_idx], dataset.test_mask[:, split_idx]

    best_h = None
    best_e_w = None
    best_epoch = -1
    best_val_metric = -1.
    best_train_metric = -1
    best_test_metric = -1.
    best_y_logits = None
    n_epochs_without_improvement = 0

    forward_time = 0
    backward_time = 0

    epoch = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        s = time.time_ns()
        _, y_prob_list, _ = model(dataset, **other_model_forward_params)
        forward_time += time.time_ns() - s

        class_loss = 0
        for y_prob in y_prob_list:
            class_loss += dataset.loss(y_prob[train_mask], y[train_mask])

        s = time.time_ns()
        class_loss.backward()
        backward_time += time.time_ns() - s

        optimiser.step()

        # eval step for validation
        model.eval()
        with th.no_grad():
            h_list, y_prob_list, e_w_list = model(dataset, **other_model_forward_params)

        y_logits = y_prob_list[-1]
        train_metric = dataset.metric(y[train_mask], y_logits[train_mask])
        val_metric = dataset.metric(y[val_mask], y_logits[val_mask])
        test_metric = dataset.metric(y[test_mask], y_logits[test_mask])

        # log info
        s = f'Epoch: {epoch}\t|\t' \
            f'Loss: {class_loss:0.4f}\t|\tTr metric: {train_metric:0.2f}\t|\tVal metric: {val_metric:0.2f}'
        txt_logger.info(s)

        if val_metric > best_val_metric:
            best_train_metric = train_metric
            best_val_metric = val_metric
            best_test_metric = test_metric
            n_epochs_without_improvement = 0
            best_epoch = epoch
            best_y_logits = y_logits
            best_h = h_list[-1]
            best_e_w = e_w_list[-1]
        else:
            n_epochs_without_improvement += 1

        if n_epochs_without_improvement == patience:
            break

    txt_logger.info(f'Best validation accuracy of {best_val_metric:0.2f} in epoch {best_epoch}')

    metrics_dict = {'training_metric': best_train_metric,
                  'validation_metric': best_val_metric,
                  'test_metric': best_test_metric,
                  'avg_forward_time': forward_time/(epoch),
                  'avg_backward_time': backward_time/(epoch)}
    return metrics_dict, best_h, best_y_logits, best_e_w


def end_to_end_training(config, split_idx, exp_dir, write_on_console):

    # load the dataset
    dataset = get_dataset(data_root=config.storage_dir, dataset_config=config.dataset_config)
    in_size = dataset.num_node_features
    out_size = dataset.num_classes

    model = create_object_from_config(config.model_config, num_in_channels=in_size, num_out_channels=out_size)
    optimiser = create_object_from_config(config.optimiser_config, params=model.parameters())

    txt_logger = get_logger(exp_dir, exp_dir, file_name='train.log', write_on_console=write_on_console)

    metrics_dict, best_h, best_y_logits, best_e_w = __train_loop__(dataset=dataset, model=model, optimiser=optimiser,
                                                                   split_idx=split_idx, txt_logger=txt_logger,
                                                                   n_epochs=config.training_config.n_epochs,
                                                                   patience=config.training_config.patience)

    for h in txt_logger.handlers:
        txt_logger.removeHandler(h)
        h.close()

    to_json_file(metrics_dict, os.path.join(exp_dir, 'training_results.json'))

    to_torch_file(best_h, os.path.join(exp_dir, 'node_embs.pt'))
    to_torch_file(best_y_logits, os.path.join(exp_dir, 'y_logits.pt'))
    to_torch_file(best_e_w, os.path.join(exp_dir, 'edge_weight.pt'))