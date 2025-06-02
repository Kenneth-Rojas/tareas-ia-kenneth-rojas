# MÓDULO inference_invertido.py: inferencia para traducción del idioma inventado al español

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model
from data_invertido import src_word2idx, tgt_word2idx, tgt_idx2word, max_len_src
import os

embedding_dim = 64
lstm_units = 128

src_vocab_size = len(src_word2idx) + 1
tgt_vocab_size = len(tgt_word2idx) + 1

# 1) Construir el modelo completo (como durante el entrenamiento)
model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense = build_model(
    src_vocab_size, tgt_vocab_size, embedding_dim, lstm_units, return_components=True
)

# 2) Cargar pesos entrenados si existen
weights_path = 'modelo_inv_es.weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print(f"Advertencia: No se encontró el archivo de pesos '{weights_path}'")

# 3) Modelo de inferencia - encoder
encoder_model = Model(encoder_inputs, encoder_states)

# 4) Modelo de inferencia - decoder
decoder_state_input_h = Input(shape=(lstm_units,), name='decoder_input_h')
decoder_state_input_c = Input(shape=(lstm_units,), name='decoder_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_states2 = [state_h2, state_c2]

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# ---- Funciones auxiliares ----

def encode_input(sentence, word2idx, max_len):
    tokens = sentence.lower().split()
    seq = [word2idx.get(word, 0) for word in tokens]
    return pad_sequences([seq], maxlen=max_len, padding='post')

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[tgt_word2idx['<start>']]])
    decoded_sentence = []

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tgt_idx2word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence) > 20:
            break

        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(decoded_sentence)
