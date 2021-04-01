import torch
import numpy as np
import scipy

from formulas_vae_v5 import evaluate_formula as my_evaluate_formula


def empirical_entropy(X):
    """
    X: np.array [n_items, n_features]
    entropy: np.array [n_items]
    """
    _, ind = torch.topk(-pairwise_dist(X, X), k=2, dim=1)
    R_i = (X - X[ind[:, 1]]).pow(2).sum(1).sqrt()
    d = X.shape[1]
    V = np.pi ** (d / 2) / scipy.special.gamma(d / 2 + 1)
    entropy = (len(R_i) * (R_i.pow(d))).log() + np.log(V) + 0.577
    return entropy

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def pick_next_point(candidate_X, X_train, y_train, model, n_sample, max_length):
    file_to_sample = 'tmp_sample_13'
    max_entropy = None
    next_point = None
    for x in candidate_X:
        # TODO(julia): decide what to do with y(x) - unknown
        cond_x = np.append(X_train, x)
        cond_y = np.append(y_train, 0)
        if len(cond_x.shape) == 1:
            cond_x = cond_x.reshape(-1, 1)
        cond_y = cond_y.reshape(-1, 1)
        cond_x = np.repeat(cond_x.reshape(1, -1, 1), n_sample, axis=0)
        cond_y = np.repeat(cond_y.reshape(1, -1, 1), n_sample, axis=0)
        with torch.no_grad():
            sample_res = model.sample(n_sample, max_length, file_to_sample, Xs=cond_x, ys=cond_y, unique=False)
        sampled_formulas, _, _, _, _ = sample_res
        _, ress, _, _ = my_evaluate_formula.evaluate_file(file_to_sample, np.append(X_train, x), np.append(y_train, 0))

        # print('ress', ress)
        # entropy = empirical_entropy(torch.tensor(ress)).mean()
        # print('entropy', entropy)

        # if max_entropy is None or max_entropy < entropy:
        #     next_point = x
            # max_entropy = entropy

        ress = torch.tensor(ress)

        var = torch.var(ress, dim=0)
        var = torch.mean(var)

        # print('var1', var)

        if max_entropy is None or max_entropy < var:
            next_point = x
            max_entropy = var

        ###### Zero sample logits
        zs = np.zeros(shape=(n_sample, model.latent_dim)).astype('f')
        with torch.no_grad():
            sample_res = model.sample(n_sample, max_length, file_to_sample, Xs=cond_x, ys=cond_y, unique=False,
                                      sample_from_logits=True, zs=zs)

        _, ress, _, _ = my_evaluate_formula.evaluate_file(file_to_sample, np.append(X_train, x), np.append(y_train, 0))

        ress = torch.tensor(ress)

        var = torch.var(ress, dim=0)
        var = torch.mean(var)

        # print('var 2', var)


    print('next point', next_point)

    return next_point, max_entropy


# if __name__ == '__main__':
#     y = torch.randn(30, 1)
#     print("Empirical entropy: {}".format(empirical_entropy(y).mean()))
#     print("Theoretical entropy: {}".format(1 / 2 + torch.tensor(3.14 * 2).sqrt().log()))

