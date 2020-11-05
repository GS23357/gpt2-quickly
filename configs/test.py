path = './dataset'
model_path = path + '/models/'

data = {
    'path': path + '/test/',
}
data['raw'] = data['path'] + 'raw.txt'
data['vocab'] = data['path'] + 'vocab.txt'
data['pickle'] = data['path'] + 'data.pickle'


model = {
    'max_length': 64,
    'n_positions': 512,
    'n_ctx': 512,
    'n_embd': 768,
    'n_layer': 6,
    'n_head': 6,
    'batch_size': 12
}


data = type('data', (), data)
model = type('model', (), model)