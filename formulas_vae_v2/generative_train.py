import torch
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error

import formulas_vae_v2.train as my_train
import formulas_vae_v2.evaluate_formula as my_evaluate_formula
import formulas_vae_v2.batch_builder as my_batch_builder


def generative_train(model, vocab, optimizer, epochs, device, batch_size,
                     n_formulas_to_sample, file_to_sample, max_length, use_for_train_fraction,
                     n_pretrain_steps, pretrain_batches, pretrain_val_batches):
    for step in range(n_pretrain_steps):
        my_train.run_epoch(vocab, model, optimizer, pretrain_batches, pretrain_val_batches, step)

    xs = np.linspace(0.0, 1.0, num=100)
    ys = 3 * xs
    for epoch in range(epochs):
        model.sample(n_formulas_to_sample, max_length, file_to_sample)
        predicted_ys = my_evaluate_formula.evaluate_file(file_to_sample, xs)
        mses = [mean_squared_error(predicted_ys[i], ys) for i in range(len(predicted_ys))]
        print(f'epoch: {epoch}, mean mse: {np.mean(mses)}')
        best_formula_indices = set(
            sorted(enumerate(mses), key=lambda x: x[1])[:int(len(mses) * use_for_train_fraction)]
        )
        best_formulas = []
        with open(file_to_sample) as f:
            for i, line in enumerate(f.readlines()):
                if i in best_formula_indices:
                    best_formulas.append(line.strip())
        with open(file_to_sample, 'w') as f:
            f.write('\n'.join(best_formulas))

        train_batches, _ = my_batch_builder.build_ordered_batches(file_to_sample, vocab, batch_size, device)
        my_train.run_epoch(vocab, model, optimizer, train_batches, pretrain_val_batches, epoch)
