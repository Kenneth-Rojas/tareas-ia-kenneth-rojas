# MÓDULO train_invertido.py: entrenamiento del modelo (idioma inventado -> español)

from data_invertido import source_tensor, decoder_input_data, decoder_target_data
from model import build_model

embedding_dim = 64
lstm_units = 128

# Tamaños de vocabulario
src_vocab_size = max(source_tensor.max(), 1) + 1
tgt_vocab_size = max(decoder_target_data.max(), 1) + 1  # usa el target real

# Construcción del modelo
model, dec_emb_layer, decoder_lstm, decoder_dense = build_model(
    src_vocab_size, tgt_vocab_size, embedding_dim, lstm_units
)

# Entrenamiento
model.fit(
    [source_tensor, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=300
)

# Guardar pesos
model.save_weights('modelo_inv_es.weights.h5')
