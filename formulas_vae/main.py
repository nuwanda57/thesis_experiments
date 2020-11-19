from torchtext import vocab, data


def main():
    src = data.Field()
    src.build_vocab(mt_train)
    Vocab.build(train_sents, vocab_file, args.vocab_size)