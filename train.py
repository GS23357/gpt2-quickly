import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import GPT2Config
from transformers import TFGPT2LMHeadModel
from transformers import XLNetTokenizer
from transformers import BertTokenizer
# from performer import PerformerConfig, TFGPT2LMHeadModel
import configs
from official import nlp
import official.nlp.optimization
import click
import time
import pickle
from pathlib import Path
import numpy as np
# from tensorflow.keras.mixed_precision import experimental as mixed_precision


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

def load_checkpoint():
    try:
        with open(f'{configs.model_path}checkpoint.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
        print('Checkpoint loaded successfully')
        return checkpoint
    except IOError:
        print('Checkpoint loading failed')
        return 

def load_tokenizer() -> XLNetTokenizer:
    tokenizer = XLNetTokenizer.from_pretrained(
        configs.data.path, max_len=configs.model.max_length, add_special_token=False)
    tokenizer.return_attention_mask = None
    return tokenizer


def get_dataset() -> tf.data.Dataset:
    p = Path(configs.data.path)
    pickle_files = p.glob('*.pickle')
    ids, labels = [], []
    for pickle_file in pickle_files:
        print(f"loading {pickle_file}")
        _ids, _labels = pickle.load(open(pickle_file, 'rb'))
        if len(ids) == 0:
            ids = _ids
            labels = _labels
        else:
            ids = np.vstack((ids, _ids))
            labels = np.vstack((labels, _labels))
    print(ids.shape, labels.shape, ids.dtype, labels.dtype)
    dataset = tf.data.Dataset.from_tensor_slices((
        ids,
        labels
    )).repeat().shuffle(ids.shape[0], reshuffle_each_iteration=True).batch(configs.model.batch_size)
    return dataset


def init_model(
    tokenizer: BertTokenizer,
    train_steps: int = 20000,
    num_warmup_steps: int = 1000,
    model_path: str = configs.model_path,
) -> TFGPT2LMHeadModel:

    try:
        model = TFGPT2LMHeadModel.from_pretrained(
            model_path, return_dict=False)
    except EnvironmentError:
        config = GPT2Config(
            architectures=["TFGPT2LMHeadModel"],
            model_type="TFGPT2LMHeadModel",
            tokenizer_class="BertTokenizer",
            vocab_size=tokenizer.vocab_size,
            n_positions=configs.model.n_positions,
            n_ctx=configs.model.n_ctx,
            n_embd=configs.model.n_embd,
            n_layer=configs.model.n_layer,
            n_head=configs.model.n_head,
            d_model=configs.model.n_embd,
            num_heads=configs.model.n_head,
            pad_token_id=tokenizer.pad_token_id,
            task_specific_params={
                "text-generation": {
                    "do_sample": True,
                    "max_length": 120
                }
            },
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        model = TFGPT2LMHeadModel(config)

    loss = model.compute_loss
    optimizer = nlp.optimization.create_optimizer(
        5e-5, num_train_steps=train_steps, 
        num_warmup_steps=num_warmup_steps,
        end_lr=1e-7,
        beta_1=0.9,
        )

    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # metric = Mymetrice('accuracy')

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        metrics=[metric]
    )
    
    optimizer._create_all_weights(model.trainable_variables)
    
    checkpoint = load_checkpoint()
    if checkpoint:
        model.optimizer.set_weights(checkpoint['opt_weights'])
        print('Optimizer loaded successfully')
    else:
        print('Using a new Optimizer')  

    return model


def train(model, train_dataset, epochs, train_steps):

    class AutoSaveCallback(tf.keras.callbacks.Callback):
        
        def on_epoch_end(self, epoch, logs=None):
            
            print('Current lr:', model.optimizer._decayed_lr('float32').numpy())
            
            self.model.save_pretrained(f'{configs.model_path}')
            
            checkpoint={'opt_weights': model.optimizer.get_weights(),
                        'epoch': epoch+1}

            with open(f'{configs.model_path}checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
                
            print(f'Checkpoint saved to {configs.model_path}')

    callbacks = [AutoSaveCallback()]
    
    checkpoint = load_checkpoint()
    if checkpoint:
        initial_epoch = checkpoint['epoch']
        print('Continue from last epoch')
    else:
        initial_epoch = 0
        print('Set initial epoch to 0')
    
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        batch_size=None,
        initial_epoch=initial_epoch
    )
    print('Training finished!')


@click.command()
@click.option('--epochs', default=50, help='number of epochs')
@click.option('--train_steps', default=1000, help='number of train_steps')
@click.option('--warmup_steps', default=300, help='number of warmup_steps')
def main(epochs, train_steps, warmup_steps):
    tokenizer = load_tokenizer()
    train_dataset = get_dataset()
    model = init_model(tokenizer, train_steps * epochs,
                       warmup_steps, configs.model_path)
    
    train(model, train_dataset, epochs, train_steps)


if __name__ == '__main__':
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        main()
