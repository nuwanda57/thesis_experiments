import numpy as np
from sklearn.metrics import mean_squared_error
from collections import deque

import formulas_vae_v2.train as my_train
import formulas_vae_v2.evaluate_formula as my_evaluate_formula
import formulas_vae_v2.batch_builder as my_batch_builder

import wandb


def generative_train(model, vocab, optimizer, epochs, device, batch_size,
                     n_formulas_to_sample, file_to_sample, max_length, percentile,
                     n_pretrain_steps, pretrain_batches, pretrain_val_batches, use_n_last_steps=6):
    for step in range(n_pretrain_steps):
        my_train.run_epoch(vocab, model, optimizer, pretrain_batches, pretrain_val_batches, step)

    xs = np.linspace(0.0, 1.0, num=100)
    ys = 3 * xs
    table = wandb.Table(columns=["correct formula"])
    table.add_data('y = 3x')
    wandb.log({'correct formula': table})
    reconstructed_formulas = []
    best_formulas = []
    best_mses = []
    last_best_sizes = deque([0] * use_n_last_steps, maxlen=use_n_last_steps)
    for epoch in range(epochs):
        s = last_best_sizes.popleft()
        best_formulas = best_formulas[s:]
        best_mses = best_mses[s:]
        wandb_log = {}
        reconstructed_formulas, _ = model.sample(n_formulas_to_sample, max_length, file_to_sample)
        predicted_ys = my_evaluate_formula.evaluate_file(file_to_sample, xs)
        mses = []
        inf = 10 ** 4
        for i in range(len(predicted_ys)):
            if predicted_ys[i] is None:
                mses.append(inf)
            else:
                mses.append(mean_squared_error(predicted_ys[i], ys))
        table = wandb.Table(columns=[f'formula, epoch: {epoch}'])
        for f in reconstructed_formulas[:20]:
            table.add_data(f)
        if (epoch + 1) % 50 == 0:
            wandb_log[f'example formulas epoch: {epoch}'] = table
        print(f'epoch: {epoch}, mean mses: {np.mean(mses)}')
        if np.isfinite(np.mean(mses)) and np.isfinite(np.log(np.mean(mses))):
            wandb_log['log_mean_mse_generated'] = np.log(np.mean(mses))
        mse_threshold = np.nanpercentile(mses + best_mses, percentile)
        epoch_best_formula_pairs = [x for x in enumerate(mses) if x[1] < mse_threshold]
        epoch_best_formula_indices = [x[0] for x in epoch_best_formula_pairs if x[1] < inf]
        epoch_best_mses = [x[1] for x in epoch_best_formula_pairs if x[1] < inf]
        print(f'epoch: {epoch}, mean best mses: {np.mean(epoch_best_mses)}')
        if np.isfinite(np.mean(epoch_best_mses)) and np.isfinite(np.log(np.mean(epoch_best_mses))):
            wandb_log['log_mean_epoch_mse_best'] = np.log(np.mean(epoch_best_mses))
        epoch_best_formulas = []
        with open(file_to_sample) as f:
            for i, line in enumerate(f.readlines()):
                if i in epoch_best_formula_indices:
                    epoch_best_formulas.append(line.strip())
        last_best_sizes.append(len(epoch_best_formulas))
        best_formulas += epoch_best_formulas
        best_mses += epoch_best_mses
        if np.isfinite(np.mean(best_mses)) and np.isfinite(np.log(np.mean(best_mses))):
            wandb_log['log_mean_mse_best'] = np.log(np.mean(best_mses))
        wandb_log['best_formulas_size'] = len(best_formulas)
        with open(file_to_sample, 'w') as f:
            f.write('\n'.join(best_formulas))
        wandb.log(wandb_log)
        if len(best_formulas) == 0:
            print('training terminated')
            break

        train_batches, _ = my_batch_builder.build_ordered_batches(file_to_sample, vocab, batch_size, device)
        my_train.run_epoch(vocab, model, optimizer, train_batches, pretrain_val_batches, epoch)
    table = wandb.Table(columns=['formula'])
    for f in reconstructed_formulas[:1000]:
        table.add_data(f)
    wandb.log({'final formula examples': table})
