import torch

import formulas_vae_v4.formula_config as my_formula_config
import formulas_vae_v4.model as my_model
import formulas_vae_v4.batch_builder as my_batch_builder
import formulas_vae_v4.generative_train as my_generative_train
import formulas_vae_v4.monitoring as my_monitoring


def exp_generative_train(xs, ys, formula, train_file, val_file, max_len, epochs,
                         model_conf_params, n_pretrain_steps=50, batch_size=256, lr=0.0005,
                         betas=(0.5, 0.999), n_formulas_to_sample=200000, percentile=20, use_n_last_steps=6,
                         project_name='experiment_generative_train', add_noise_to_model_params=False,
                         noise_to_model_params_weight=0.01, add_noise_every_n_steps=1, no_retrain=False,
                         continue_training_on_train_dataset=False):

    experiment_config = {
        'max_len': max_len, 'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr,
        'n_formulas_sampled': n_formulas_to_sample, 'percentile': percentile,
        'n_pretrain_steps': n_pretrain_steps, 'use_n_last_steps': use_n_last_steps,
        'no_retrain': int(no_retrain), 'continue_training_on_train_dataset': int(continue_training_on_train_dataset),
        'add_noise_to_model_params': int(add_noise_to_model_params),
        'noise_to_model_params_weight': int(noise_to_model_params_weight),
        'add_noise_every_n_steps': int(add_noise_every_n_steps),
    }
    monitoring = my_monitoring.Monitoring(project_name=project_name, correct_formula=formula,
                                          x_range=f'x_min={min(xs)}, x_max={max(xs)}, x_count={len(xs)}',
                                          experiment_config=experiment_config)

    device = torch.device('cuda')
    train_batches, _ = my_batch_builder.build_ordered_batches(formula_file=train_file, batch_size=batch_size,
                                                              device=device)
    valid_batches, _ = my_batch_builder.build_ordered_batches(formula_file=val_file, batch_size=batch_size,
                                                              device=device)
    model_params = my_model.ModelParams(vocab_size=len(my_formula_config.INDEX_TO_TOKEN), device=device, **model_conf_params)
    model = my_model.FormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    my_generative_train.generative_train(model=model, optimizer=optimizer, epochs=epochs, device=device,
                                         batch_size=batch_size, n_formulas_to_sample=n_formulas_to_sample,
                                         file_to_sample='sample', max_length=max_len, percentile=percentile,
                                         n_pretrain_steps=n_pretrain_steps, pretrain_batches=train_batches,
                                         pretrain_val_batches=valid_batches, xs=xs, ys=ys, formula=formula,
                                         use_n_last_steps=use_n_last_steps, monitoring=monitoring,
                                         add_noise_to_model_params=add_noise_to_model_params,
                                         noise_to_model_params_weight=noise_to_model_params_weight,
                                         add_noise_every_n_steps=add_noise_every_n_steps, no_retrain=no_retrain,
                                         continue_training_on_train_dataset=continue_training_on_train_dataset)
