import os
import numpy as np

import torch

from sklearn.metrics import mean_squared_error

import formulas_vae_v2.vocab as my_vocab
import formulas_vae_v2.model as my_model
import formulas_vae_v2.train as my_train
import formulas_vae_v2.batch_builder as my_batch_builder
import formulas_vae_v2.generative_train as my_generative_train
import formulas_vae_v2.evaluate_formula as my_evaluate_formula

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
        model_conf_params, n_pretrain_steps=50, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
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
    # wandb_log = {
    #     'max_len': max_len,
    #     'epochs': epochs,
    #     'batch_size': batch_size,
    #     'learning_rate': lr,
    #     'n_formulas_sampled': n_formulas_to_sample,
    #     'chosen_for_train_fraction': use_for_train_fraction,
    #     'n_pretrain_steps': n_pretrain_steps,
    # }
    table = wandb.Table(columns=["max_len", "epochs", "batch_size", "learning_rate", "n_formulas_sampled",
                                 "chosen_for_train_fraction", "n_pretrain_steps"])
    table.add_data(max_len, epochs, batch_size, lr, n_formulas_to_sample, use_for_train_fraction, n_pretrain_steps)
    wandb.log({'configs': table})
    my_generative_train.generative_train(model, vocab, optimizer, epochs, device, batch_size,
                                         n_formulas_to_sample, 'sample', max_len, use_for_train_fraction,
                                         n_pretrain_steps, train_batches, valid_batches)


def exp_check_no_results(train_file, val_file, test_file, reconstruct_strategy, max_len, epochs, results_dir,
        model_conf_params, n_pretrain_steps=50, batch_size=256, lr=0.0005, betas=(0.5, 0.999)):
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
    for step in range(n_pretrain_steps):
        my_train.run_epoch(vocab, model, optimizer, train_batches, valid_batches, step)

    xs = np.linspace(0.0, 1.0, num=100)
    ys = 3 * xs
    epoch_best = []
    for epoch in range(1500):
        reconstructed_formulas, _ = model.sample(n_formulas_to_sample, max_len, 'sample')
        predicted_ys = my_evaluate_formula.evaluate_file('sample', xs)
        mses = []
        inf = 10 ** 4
        for i in range(len(predicted_ys)):
            if predicted_ys[i] is None:
                mses.append(inf)
            else:
                try:
                    mses.append(mean_squared_error(predicted_ys[i], ys))
                except:
                    print('exception when calculating mse')
                    print(predicted_ys[i][:100])
                    mses.append(inf)
        best_formula_pairs = sorted(enumerate(mses), key=lambda x: x[1])[:int(len(mses) * use_for_train_fraction)]
        best_formula_pairs = [x for x in best_formula_pairs if x[1] < inf]
        best_formula_mses = [x[1] for x in best_formula_pairs]
        epoch_best += best_formula_mses
        epoch_best = sorted(epoch_best)[:400]
        print(f'{epoch} mean best mses: {np.mean(epoch_best)}')
        print(f'{epoch} mean best mses log : {np.log(epoch_best)}')

    best_formula_mses = sorted(epoch_best)[:400]
    print(f'mean best mses: {np.mean(best_formula_mses)}')
    print(f'mean best mses log : {np.log(np.mean(best_formula_mses))}')
    # if np.isfinite(np.mean(mses)) and np.isfinite(np.log(np.mean(mses))):
    #     print(f'mean mses : {np.mean(mses)}')
    #     print(f'mean mses log : {np.log(np.mean(mses))}')
