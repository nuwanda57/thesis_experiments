import numpy as np
from sklearn.metrics import mean_squared_error

import formulas_vae_v2.train as my_train
import formulas_vae_v2.evaluate_formula as my_evaluate_formula
import formulas_vae_v2.batch_builder as my_batch_builder

import wandb


def generative_train(model, vocab, optimizer, epochs, device, batch_size,
                     n_formulas_to_sample, file_to_sample, max_length, use_for_train_fraction,
                     n_pretrain_steps, pretrain_batches, pretrain_val_batches):
    for step in range(n_pretrain_steps):
        my_train.run_epoch(vocab, model, optimizer, pretrain_batches, pretrain_val_batches, step)

    xs = np.linspace(0.0, 1.0, num=100)
    ys = 3 * xs
    table = wandb.Table(columns=["correct formula"])
    table.add_data('y = 3x')
    wandb.log({'correct formula': table})
    reconstructed_formulas = []
    for epoch in range(epochs):
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
        wandb_log['example formulas'] = table
        print(f'epoch: {epoch}, mean mses: {np.mean(mses)}')
        if np.isfinite(np.mean(mses)):
            wandb_log['mean_mse_generated'] = np.mean(mses)
        best_formula_pairs = sorted(enumerate(mses), key=lambda x: x[1])[:int(len(mses) * use_for_train_fraction)]
        best_formula_pairs = [x for x in best_formula_pairs if x[1] < inf]
        best_formula_mses = [x[1] for x in best_formula_pairs]
        print(f'epoch: {epoch}, mean best mses: {np.mean(best_formula_mses)}')
        if np.isfinite(np.mean(best_formula_mses)):
            wandb_log['mean_mse_best_formulas'] = np.mean(best_formula_mses)
        wandb.log(wandb_log)
        best_formula_indices = [x[0] for x in best_formula_pairs]
        best_formulas = []
        with open(file_to_sample) as f:
            for i, line in enumerate(f.readlines()):
                if i in best_formula_indices:
                    best_formulas.append(line.strip())
        with open(file_to_sample, 'w') as f:
            f.write('\n'.join(best_formulas))

        train_batches, _ = my_batch_builder.build_ordered_batches(file_to_sample, vocab, batch_size, device)
        my_train.run_epoch(vocab, model, optimizer, train_batches, pretrain_val_batches, epoch)
    table = wandb.Table(columns=['formula'])
    for f in reconstructed_formulas[:1000]:
        table.add_data(f)
    wandb.log({'final formula examples': table})
