import os
import numpy as np

import torch

import formulas_vae_v2.vocab as my_vocab
import formulas_vae_v2.model as my_model
import formulas_vae_v2.train as my_train
import formulas_vae_v2.batch_builder as my_batch_builder


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
