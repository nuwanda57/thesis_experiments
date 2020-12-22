import os
import json
import numpy as np

import torch

import formulas_vae.vocab as my_vocab
import formulas_vae.utils as my_utils
import formulas_vae.model as my_model
import formulas_vae.train as my_train
import formulas_vae.train_best_worst as my_train_best_worst

import results.analyse_results as my_analyse_results


def reconstruct_test_based_on_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
    device = torch.device('cuda')
    train_batches, _ = my_utils.build_ordered_batches(train_file, vocab, batch_size, device)
    valid_batches, _ = my_utils.build_ordered_batches(val_file, vocab, batch_size, device)
    test_batches, test_order = my_utils.build_ordered_batches(test_file, vocab, batch_size, device)
    rec_file_template = os.path.join(results_dir, 'rec_%d')

    for epochs in epochs_list:
        if os.path.exists(rec_file_template % epochs):
            print('WARNING: rec file already exists, skipping epochs %d' % epochs)
            continue
        model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
        model = my_model.ExtendedFormulaVARE(model_params)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        my_train.train(vocab, model, optimizer, train_batches, valid_batches, epochs)
        model.reconstruct(test_batches, test_order, max_len, rec_file_template % epochs, strategy=reconstruct_strategy)


def reconstruct_test_based_on_epoch_tmp(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
    device = torch.device('cuda')
    valid_batches, _ = my_utils.build_ordered_batches(val_file, vocab, batch_size, device)
    test_batches, test_order = my_utils.build_ordered_batches(test_file, vocab, batch_size, device)
    rec_file_template = os.path.join(results_dir, 'rec_%d')

    for epochs in epochs_list:
        if os.path.exists(rec_file_template % epochs):
            print('WARNING: rec file already exists, skipping epochs %d' % epochs)
            continue
        model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
        model = my_model.ExtendedFormulaVARE(model_params)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        my_train_best_worst.train(
            vocab, model, optimizer, train_file, valid_batches, epochs, batch_size, max_len, device,
            log_interval=20, update_train_epochs=50, training_log_dir='training/', choose_worst=True)
        model.reconstruct(test_batches, test_order, max_len, rec_file_template % epochs, strategy=reconstruct_strategy)


def percent_of_reconstructed_formulas_based_depending_on_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    reconstruct_test_based_on_epoch_tmp(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=batch_size, lr=lr, betas=betas)

    stats = []
    rec_file_template = os.path.join(results_dir, 'rec_%d')
    for epochs in epochs_list:
        _, _, percent_correct = my_analyse_results.main(rec_file_template % epochs, test_file)
        stats.append(percent_correct)

    stats_file = os.path.join(results_dir, 'stats.json')
    with open(stats_file, 'w') as outfile:
        json.dump(stats, outfile)

    return stats


def mse_on_reconstructed_formulas_based_depending_on_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    reconstruct_test_based_on_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs_list, results_dir,
        model_conf_params, batch_size=batch_size, lr=lr, betas=betas)

    stats = []
    rec_file_template = os.path.join(results_dir, 'rec_%d')
    for epochs in epochs_list:
        stats.append(my_utils.mean_reconstruction_mse(rec_file_template % epochs, test_file, [0.1, 0.2, 0.5]))

    stats_file = os.path.join(results_dir, 'stats.json')
    with open(stats_file, 'w') as outfile:
        json.dump(stats, outfile)

    return stats