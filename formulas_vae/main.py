from formulas_vae import vocab as my_vocab, model as my_model
from formulas_vae import utils as my_utils
from formulas_vae import train as my_train
from formulas_vae import evaluate as my_evaluate

import torch


def main(train_file, val_file, test_file, epochs, log_period, out_file):
    vocab = my_vocab.Vocab()
    vocab.build_from_formula_file(train_file)
    vocab.write_vocab_to_file('vocab.txt')
    device = torch.device('cuda')
    train_batches, _ = my_utils.build_ordered_batches(train_file, vocab, 256, device)
    valid_batches, _ = my_utils.build_ordered_batches(val_file, vocab, 256, device)
    test_batches, test_order = my_utils.build_ordered_batches(test_file, vocab, 256, device)

    model_params = my_model.ModelParams(
        vocab=vocab, device=device, vocab_size=vocab.size(), token_embedding_dim=128, hidden_dim=128,
        encoder_layers_cnt=1, decoder_layers_cnt=1, latent_dim=8
    )
    model = my_model.ExtendedFormulaVARE(model_params)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
    my_train.train(vocab, model, optimizer, train_batches, valid_batches, epochs, log_interval=log_period)
    model.reconstruct(test_batches, test_order, 50, out_file)
    my_evaluate.reconstruct(vocab, device, model, test_batches, test_order, 50, 'test_out')


if __name__ == '__main__':
    main('formulas_train_5_10.txt', 'formulas_val_5_10.txt', 'formulas_test_5_10.txt', 100, 20, 'out')
