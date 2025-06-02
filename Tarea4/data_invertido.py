# MÓDULO data_invertido.py: datos, tokenización y preparación de secuencias (idioma inventado -> español)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Invertimos las oraciones
source_sentences = [
    'bönjúr kömmó étás',
    'mèrsì për lèvnìr',
    'búèn dìàs mönsïè',
    'órëvwàr è bón sòrt',
    'òn së vòà dèmän',
    'bönjúr',
    'mèrsì',
    'búèn dìàs',
    'órëvwàr',
    'òn së vòà',
    'pär fävór',
    'lö siéntö',
    'té kíéro',
    'félíz kùmplëánjòs',
    'búèn träbájó'
]

target_sentences = [
    '<start> hola cómo estás <end>',
    '<start> gracias por venir <end>',
    '<start> buenos días señor <end>',
    '<start> adiós y buena suerte <end>',
    '<start> nos vemos mañana <end>',
    '<start> hola <end>',
    '<start> gracias <end>',
    '<start> buenos días <end>',
    '<start> adiós <end>',
    '<start> nos vemos <end>',
    '<start> por favor <end>',
    '<start> lo siento <end>',
    '<start> te quiero <end>',
    '<start> feliz cumpleaños <end>',
    '<start> buen trabajo <end>'
]

# Tokenizadores
src_tokenizer = Tokenizer(filters='', lower=True)
tgt_tokenizer = Tokenizer(filters='', lower=True)

src_tokenizer.fit_on_texts(source_sentences)
tgt_tokenizer.fit_on_texts(target_sentences)

src_word2idx = src_tokenizer.word_index
tgt_word2idx = tgt_tokenizer.word_index
tgt_idx2word = {i: w for w, i in tgt_word2idx.items()}

# Secuencias tokenizadas
source_seqs = src_tokenizer.texts_to_sequences(source_sentences)
target_seqs = tgt_tokenizer.texts_to_sequences(target_sentences)

# Longitudes máximas
max_len_src = max(len(seq) for seq in source_seqs)
max_len_tgt = max(len(seq) for seq in target_seqs)

# Padding para encoder
source_tensor = pad_sequences(source_seqs, maxlen=max_len_src, padding='post')

# División en input y output para el decoder
decoder_input_seqs = [seq[:-1] for seq in target_seqs]  # sin <end>
decoder_target_seqs = [seq[1:] for seq in target_seqs]  # sin <start>

decoder_input_data = pad_sequences(decoder_input_seqs, maxlen=max_len_tgt - 1, padding='post')
decoder_target_data = pad_sequences(decoder_target_seqs, maxlen=max_len_tgt - 1, padding='post')
decoder_target_data = np.expand_dims(decoder_target_data, -1)  # requerido por sparse_categorical_crossentropy

__all__ = [
    'source_tensor', 'decoder_input_data', 'decoder_target_data',
    'src_word2idx', 'tgt_word2idx', 'tgt_idx2word',
    'max_len_src', 'max_len_tgt'
]
