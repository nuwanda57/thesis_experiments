import os
import numpy as np

import torch

from sklearn.metrics import mean_squared_error

import formulas_vae_v4.formula_config as my_formula_config
import formulas_vae_v4.model as my_model
import formulas_vae_v4.train as my_train
import formulas_vae_v4.batch_builder as my_batch_builder
import formulas_vae_v4.generative_train as my_generative_train
import formulas_vae_v4.evaluate_formula as my_evaluate_formula

import wandb


def exp_generative_train(xs, ys, formula, train_file, val_file, test_file, reconstruct_strategy, max_len, epochs,
                         results_dir, model_conf_params, n_pretrain_steps=50, batch_size=256, lr=0.0005,
                         betas=(0.5, 0.999), n_formulas_to_sample=200000, percentile=20, use_n_last_steps=6,
                         do_sample_unique=False):
    wandb.init(project="generative train no constants")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    training_log_dir = os.path.join(results_dir, 'training/')
    if not os.path.exists(training_log_dir):
        os.mkdir(training_log_dir)

    device = torch.device('cuda')
    # train_batches, _ = my_batch_builder.build_ordered_batches(train_file, batch_size, device)
    # valid_batches, _ = my_batch_builder.build_ordered_batches(val_file, batch_size, device)

    model_params = my_model.ModelParams(vocab_size=len(my_formula_config.INDEX_TO_TOKEN), device=device, **model_conf_params)
    model = my_model.FormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    table = wandb.Table(columns=["max_len", "epochs", "batch_size", "learning_rate", "n_formulas_sampled",
                                 "chosen_for_train_fraction", "n_pretrain_steps", "use_n_last_steps"])
    table.add_data(max_len, epochs, batch_size, lr, n_formulas_to_sample, percentile, n_pretrain_steps, use_n_last_steps)
    wandb.log({'configs': table})
    my_generative_train.generative_train(model, optimizer, epochs, device, batch_size,
                                         n_formulas_to_sample, 'sample', max_len, percentile,
                                         n_pretrain_steps, None, None, xs, ys,
                                         formula, use_n_last_steps, do_sample_unique)
