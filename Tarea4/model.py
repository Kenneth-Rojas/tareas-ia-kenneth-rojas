from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

def build_model(src_vocab_size, tgt_vocab_size, embedding_dim=64, lstm_units=128, return_components=False):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(input_dim=src_vocab_size, output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True, name='encoder_lstm')(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb_layer = Embedding(input_dim=tgt_vocab_size, output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(tgt_vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Modelo completo para entrenamiento
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if return_components:
        return model, encoder_inputs, encoder_states, decoder_inputs, dec_emb_layer, decoder_lstm, decoder_dense
    else:
        return model, dec_emb_layer, decoder_lstm, decoder_dense
