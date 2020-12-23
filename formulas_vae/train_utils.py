import os

import formulas_vae.utils as my_utils


def update_train_dataset(do_update, old_train_path, new_train_path, vocab, model, batch_size, device, max_len,
                         xs, training_dir, choose_worst, fraction):
    if not do_update:
        return
    train_batches, train_order = my_utils.build_ordered_batches(old_train_path, vocab, batch_size, device)
    rec_path = os.path.join(training_dir, 'rec')
    model.reconstruct(train_batches, train_order, max_len, rec_path, strategy='mu')
    mses = my_utils.reconstruction_mses(rec_path, old_train_path, xs)

    mse_with_inds = list(enumerate(mses))
    mse_with_inds.sort(key=lambda k: k[1], reverse=choose_worst)

    new_formula_indices = set([m[0] for m in mse_with_inds[:(int(len(mse_with_inds) * fraction))]])
    written = 0
    with open(old_train_path) as old_file, open(new_train_path, 'w') as new_file:
        for i, line1 in enumerate(old_file):
            line = line1.strip()
            if i in new_formula_indices:
                new_file.write(line)
                written += 1
                if written != len(new_formula_indices):
                    new_file.write('\n')
