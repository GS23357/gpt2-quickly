import sentencepiece as spm
import configs
import click
import os


@click.command()
@click.option('--vocab_size', default=32000, help='number of vocabs')
@click.option('--character_coverage', default=0.999, help='percentage of characters included')
def train_with_sentenceprices(vocab_size, character_coverage, num_threads=2):
    spm.SentencePieceTrainer.train(input=configs.data.raw_cut, 
                                   model_prefix='spiece',
                                   model_type='bpe',
                                   character_coverage=character_coverage,
                                   vocab_size=vocab_size,
                                   num_threads=num_threads,
                                   pad_id=0,
                                   unk_id=3,
                                   )
    os.system(f"mv spiece.model {configs.data.path}")


if __name__ == '__main__':
    train_with_sentenceprices()
