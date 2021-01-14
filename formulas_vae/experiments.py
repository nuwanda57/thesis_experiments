import os
import numpy as np

import torch

import formulas_vae.vocab as my_vocab
import formulas_vae.utils as my_utils
import formulas_vae.model as my_model
import formulas_vae.train as my_train
import formulas_vae.train_utils as my_train_utils


def reconstruct_test_per_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs, results_dir,
        model_conf_params, train_dataset_update_frequency,
        train_dataset_update=False, choose_worst=None, fraction=None,
        reconstruct_frequency=50, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    training_log_dir = os.path.join(results_dir, 'training/')
    if not os.path.exists(training_log_dir):
        os.mkdir(training_log_dir)
    old_train_file = train_file
    training_new_file_template = os.path.join(training_log_dir, 'train_%d.txt')

    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
    device = torch.device('cuda')
    train_batches, _ = my_utils.build_ordered_batches(train_file, vocab, batch_size, device)
    valid_batches, _ = my_utils.build_ordered_batches(val_file, vocab, batch_size, device)
    test_batches, test_order = my_utils.build_ordered_batches(test_file, vocab, batch_size, device)
    rec_file_template = os.path.join(results_dir, 'rec_%d')

    model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
    model = my_model.ExtendedFormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    for epoch in range(1, epochs + 1):
        my_train.run_epoch(vocab, model, optimizer, train_batches, valid_batches, epoch)
        if epoch % reconstruct_frequency == 0:
            model.reconstruct(
                test_batches, test_order, max_len, rec_file_template % epoch, strategy=reconstruct_strategy)
        if epoch % train_dataset_update_frequency == 0 and train_dataset_update:
            xs = np.linspace(0.1, 1.0, num=10)
            new_train_file = training_new_file_template % epoch
            my_train_utils.update_train_dataset(train_dataset_update, old_train_file, new_train_file, vocab, model,
                                                batch_size, device, max_len, xs, training_log_dir, choose_worst,
                                                fraction)
            train_batches, _ = my_utils.build_ordered_batches(new_train_file, vocab, batch_size, device)
            old_train_file = new_train_file


# def reconstruct_test_based_on_epoch_tmp(
#         train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
#         model_conf_params, update_train_epochs, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
#     if not os.path.exists(results_dir):
#         os.mkdir(results_dir)
#
#     vocab = my_vocab.Vocab()
#     vocab.build_from_formula_file(train_file)
#     vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
#     device = torch.device('cuda')
#     valid_batches, _ = my_utils.build_ordered_batches(val_file, vocab, batch_size, device)
#     test_batches, test_order = my_utils.build_ordered_batches(test_file, vocab, batch_size, device)
#     rec_file_template = os.path.join(results_dir, 'rec_%d')
#
#     for epochs in epochs_list:
#         if os.path.exists(rec_file_template % epochs):
#             print('WARNING: rec file already exists, skipping epochs %d' % epochs)
#             continue
#         model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
#         model = my_model.ExtendedFormulaVARE(model_params)
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
#         my_train_best_worst.train(
#             vocab, model, optimizer, train_file, valid_batches, epochs, batch_size, max_len, device,
#             log_interval=20, update_train_epochs=update_train_epochs, training_log_dir='training/', choose_worst=True)
#         model.reconstruct(test_batches, test_order, max_len, rec_file_template % epochs, strategy=reconstruct_strategy)
#
#
# def percent_of_reconstructed_formulas_based_depending_on_epoch(
#         train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
#         model_conf_params, update_train_epochs, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
#
#     if not os.path.exists(results_dir):
#         os.mkdir(results_dir)
#
#     reconstruct_test_based_on_epoch_tmp(
#         train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
#         model_conf_params, update_train_epochs, batch_size=batch_size, lr=lr, betas=betas)
#
#     stats = []
#     rec_file_template = os.path.join(results_dir, 'rec_%d')
#     for epochs in epochs_list:
#         _, _, percent_correct = my_analyse_results.main(rec_file_template % epochs, test_file)
#         stats.append(percent_correct)
#
#     stats_file = os.path.join(results_dir, 'stats.json')
#     with open(stats_file, 'w') as outfile:
#         json.dump(stats, outfile)
#
#     return stats
#
#
# def mse_on_reconstructed_formulas_based_depending_on_epoch(
#         train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
#         model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
#     if not os.path.exists(results_dir):
#         os.mkdir(results_dir)
#
#     reconstruct_test_based_on_epoch(
#         train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
#         model_conf_params, batch_size=batch_size, lr=lr, betas=betas)
#
#     stats = []
#     rec_file_template = os.path.join(results_dir, 'rec_%d')
#     for epochs in epochs_list:
#         stats.append(my_utils.mean_reconstruction_mse(rec_file_template % epochs, test_file, [0.1, 0.2, 0.5]))
#
#     stats_file = os.path.join(results_dir, 'stats.json')
#     with open(stats_file, 'w') as outfile:
#         json.dump(stats, outfile)
#
#     return stats
