import formulas_vae_v4.train as my_train
import formulas_vae_v4.evaluate_formula as my_evaluate_formula
import formulas_vae_v4.formula_utils as my_formula_utils
import formulas_vae_v4.batch_builder as my_batch_builder

import torch

import numpy as np
from collections import deque

import wandb


def _pretrain(n_pretrain_steps, model, optimizer, pretrain_batches, pretrain_val_batches):
    for step in range(n_pretrain_steps):
        my_train.run_epoch(model, optimizer, pretrain_batches, pretrain_val_batches, step)


def log_mses_wandb(sorted_best_mses, sorted_best_formulas, wandb_log, epoch, prefix):
    wandb_log[f'{prefix}_best_formulas_size'] = len(sorted_best_formulas)
    if np.isfinite(np.mean(sorted_best_mses)):
        if np.mean(sorted_best_mses) != 0:
            wandb_log[f'{prefix}_log_mean_mse_all'] = np.log(np.mean(sorted_best_mses))
        else:
            wandb_log[f'{prefix}_log_mean_mse_all'] = -100
        for count in [1, 10, 25, 50, 100, 200, 400]:
            if len(sorted_best_mses) < count:
                continue
            if np.mean(sorted_best_mses[:count]) != 0:
                wandb_log[f'{prefix}_log_mean_mse_top_{count}'] = np.log(np.mean(sorted_best_mses[:count]))
            else:
                wandb_log[f'{prefix}_log_mean_mse_top_{count}'] = -100
        if (epoch + 1) % 50 == 0:
            table = wandb.Table(columns=[f'{prefix}_best formulas, epoch: {epoch}', 'mse'])
            for f, m in zip(sorted_best_formulas[:20], sorted_best_mses[:20]):
                table.add_data(my_formula_utils.get_formula_representation(f.split()), str(m))
            wandb_log[f'{prefix}_example formulas epoch: {epoch}'] = table


class Statistics:
    def __init__(self, use_n_last_steps, percentile):
        self.reconstructed_formulas = []
        self.last_n_best_formulas = []
        self.last_n_best_mses = []
        self.last_n_best_sizes = deque([0] * use_n_last_steps, maxlen=use_n_last_steps)
        self.the_best_formulas = []
        self.the_best_mses = []
        self.percentile = percentile

    def clear_the_oldest_step(self):
        s = self.last_n_best_sizes.popleft()
        self.last_n_best_formulas = self.last_n_best_formulas[s:]
        self.last_n_best_mses = self.last_n_best_mses[s:]

    def save_best_samples(self, sampled_mses, sampled_formulas, wandb_log, epoch):
        mse_threshold = np.nanpercentile(sampled_mses + self.last_n_best_mses, self.percentile)
        epoch_best_mses = [x for x in sampled_mses if x < mse_threshold]
        epoch_best_formulas = [
            sampled_formulas[i] for i in range(len(sampled_formulas)) if sampled_mses[i] < mse_threshold]
        assert len(epoch_best_mses) == len(epoch_best_formulas)
        wandb_log['mean_mse_sampled'] = np.mean(sampled_mses)
        wandb_log[f'mean_mse_sampled_{self.percentile}_percentile'] = np.mean(epoch_best_mses)
        wandb_log['formulas_sampled_count'] = len(sampled_mses)
        wandb_log[f'formulas_sampled_{self.percentile}_percentile_count'] = len(epoch_best_mses)

        self.last_n_best_sizes.append(len(epoch_best_formulas))
        self.last_n_best_mses += epoch_best_mses
        self.last_n_best_formulas += epoch_best_formulas
        self._update_the_best_formulas(epoch_best_formulas=epoch_best_formulas, epoch_best_mses=epoch_best_mses)
        log_mses_wandb(self.the_best_mses, self.the_best_formulas, wandb_log, epoch, 'the_best')

        sorted_epoch_best_pairs = sorted(zip(epoch_best_mses, epoch_best_formulas))
        sorted_epoch_best_formulas = [x[1] for x in sorted_epoch_best_pairs]
        sorted_epoch_best_mses = [x[0] for x in sorted_epoch_best_pairs]
        log_mses_wandb(sorted_epoch_best_mses, sorted_epoch_best_formulas, wandb_log, epoch, 'sampled')

    def _update_the_best_formulas(self, epoch_best_formulas, epoch_best_mses):
        self.the_best_formulas += epoch_best_formulas
        self.the_best_mses += epoch_best_mses

        the_best_pairs = sorted(zip(self.the_best_mses, self.the_best_formulas))[:200]
        used_formulas = set()
        self.the_best_formulas = []
        self.the_best_mses = []
        for i in range(len(the_best_pairs)):
            if the_best_pairs[i][1] not in used_formulas:
                self.the_best_formulas.append(the_best_pairs[i][1])
                self.the_best_mses.append(the_best_pairs[i][0])
            used_formulas.add(the_best_pairs[i][1])

    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.last_n_best_formulas))


def generative_train(model, optimizer, epochs, device, batch_size,
                     n_formulas_to_sample, file_to_sample, max_length, percentile,
                     n_pretrain_steps, pretrain_batches, pretrain_val_batches, xs,
                     ys, formula, use_n_last_steps, monitoring, add_noise_to_model_params=False,
                     noise_to_model_params_weight=0.01):
    _pretrain(n_pretrain_steps, model, optimizer, pretrain_batches, pretrain_val_batches)

    retrain_file = f'{file_to_sample}-train'
    stats = Statistics(use_n_last_steps=use_n_last_steps, percentile=percentile)
    for epoch in range(epochs):
        wandb_log = {}
        stats.clear_the_oldest_step()

        if add_noise_to_model_params:
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn(param.size()).to(device) * 0.01 * torch.norm(param).to(device))

        sample_res = model.sample(n_formulas_to_sample, max_length, file_to_sample)
        sampled_formulas, zs, n_formulas_sampled, n_valid_formulas_sampled, n_unique_valid_formulas_sampled = sample_res
        wandb_log['n_formulas_sampled'] = n_formulas_sampled
        wandb_log['n_valid_formulas_sampled'] = n_valid_formulas_sampled
        wandb_log['n_unique_valid_formulas_sampled'] = n_unique_valid_formulas_sampled

        sampled_formulas = [' '.join(f) for f in sampled_formulas]
        mses, ress, coeffs, optimized_formulas = my_evaluate_formula.evaluate_file(file_to_sample, xs, ys)
        stats.save_best_samples(sampled_mses=mses, sampled_formulas=sampled_formulas, wandb_log=wandb_log, epoch=epoch)

        stats.write_last_n_to_file(retrain_file)

        monitoring.log(wandb_log)
        train_batches, _ = my_batch_builder.build_ordered_batches(retrain_file, batch_size, device)
        my_train.run_epoch(model, optimizer, train_batches, train_batches, epoch)
