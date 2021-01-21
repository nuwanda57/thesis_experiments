import os
import numpy as np

import torch

import formulas_vae_v2.vocab as my_vocab
import formulas_vae_v2.model as my_model
import formulas_vae_v2.train as my_train
import formulas_vae_v2.batch_builder as my_batch_builder
import formulas_vae_v2.generative_train as my_generative_train

import wandb


def reconstruct_test_per_epoch(
        train_file, val_file, test_file, reconstruct_strategy, max_len, epochs, results_dir,
        model_conf_params, reconstruct_frequency=50, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    training_log_dir = os.path.join(results_dir, 'training/')
    if not os.path.exists(training_log_dir):
        os.mkdir(training_log_dir)

    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
    device = torch.device('cuda')
    train_batches, _ = my_batch_builder.build_ordered_batches(train_file, vocab, batch_size, device)
    valid_batches, _ = my_batch_builder.build_ordered_batches(val_file, vocab, batch_size, device)
    test_batches, test_order = my_batch_builder.build_ordered_batches(test_file, vocab, batch_size, device)
    rec_file_template = os.path.join(results_dir, 'rec_%d')

    model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
    model = my_model.FormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    for epoch in range(1, epochs + 1):
        my_train.run_epoch(vocab, model, optimizer, train_batches, valid_batches, epoch)
        if epoch % reconstruct_frequency == 0:
            model.reconstruct(
                test_batches, test_order, max_len, rec_file_template % epoch, strategy=reconstruct_strategy)


def exp_generative_train(train_file, val_file, test_file, reconstruct_strategy, max_len, epochs, results_dir,
        model_conf_params, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
    wandb.init(project="generative train")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    training_log_dir = os.path.join(results_dir, 'training/')
    if not os.path.exists(training_log_dir):
        os.mkdir(training_log_dir)

    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file(os.path.join(results_dir, 'vocab.txt'))
    device = torch.device('cuda')
    train_batches, _ = my_batch_builder.build_ordered_batches(train_file, vocab, batch_size, device)
    valid_batches, _ = my_batch_builder.build_ordered_batches(val_file, vocab, batch_size, device)

    model_params = my_model.ModelParams(vocab=vocab, vocab_size=vocab.size(), device=device, **model_conf_params)
    model = my_model.FormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    n_formulas_to_sample = 2000
    use_for_train_fraction = 0.2
    n_pretrain_steps = 250
    wandb_log = {
        'max_len': max_len,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'n_formulas_sampled': n_formulas_to_sample,
        'chosen_for_train_fraction': use_for_train_fraction,
        'n_pretrain_steps': n_pretrain_steps,
    }
    wandb.log(wandb_log)
    my_generative_train.generative_train(model, vocab, optimizer, epochs, device, batch_size,
                                         n_formulas_to_sample, 'sample', max_len, use_for_train_fraction,
                                         n_pretrain_steps, train_batches, valid_batches)
