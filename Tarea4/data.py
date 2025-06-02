# MÓDULO data.py: datos, tokenización y preparación de secuencias

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

source_sentences = [
    'hola cómo estás',
    'gracias por venir',
    'buenos días señor',
    'adiós y buena suerte',
    'nos vemos mañana',
    'hola',
    'gracias',
    'buenos días',
    'adiós',
    'nos vemos',
    'por favor',
    'lo siento',
    'te quiero',
    'feliz cumpleaños',
    'buen trabajo'
]

target_sentences = [
    '<start> bönjúr kömmó étás <end>',
    '<start> mérsì për lèvnìr <end>',
    '<start> búèn dìàs mönsïè <end>',
    '<start> órëvwàr è bón sòrt <end>',
    '<start> òn së vòà dèmän <end>',
    '<start> bönjúr <end>',
    '<start> mérsì <end>',
    '<start> búèn dìàs <end>',
    '<start> órëvwàr <end>',
    '<start> òn së vòà <end>',
    '<start> pär fävór <end>',
    '<start> lö siéntö <end>',
    '<start> té kíéro <end>',
    '<start> félíz kùmplëánjòs <end>',
    '<start> búèn träbájó <end>'
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

__all__ = ['source_tensor', 'decoder_input_data', 'decoder_target_data']
