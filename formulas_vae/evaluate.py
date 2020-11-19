import formulas_vae.utils as my_utils


def reconstruct(model, vocab, formulas_file, device):
    formulas = []
    with open(formulas_file) as f:
        for line in f:
            formulas.append(line.split())

    print(formulas[:3])
    batch = my_utils.build_single_batch_from_formulas_list(formulas[:3], vocab, device)

    _, _, _, z = model(batch)

    answers = []
    reconstructed_inds = model.generate_greedy(z, 10).view(-1, 10)
    print(reconstructed_inds)
    for ri in reconstructed_inds:
        reconstructed = [vocab.index_to_token[i] for i in ri]
        answers.append(reconstructed)

    return answers
