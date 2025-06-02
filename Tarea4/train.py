# MÓDULO train.py: entrenamiento del modelo

from data import source_tensor, decoder_input_data, decoder_target_data
from model import build_model

embedding_dim = 64
lstm_units = 128

src_vocab_size = max(source_tensor.max(), 1) + 1
tgt_vocab_size = max(decoder_target_data.max(), 1) + 1  # usa el target real

model, dec_emb_layer, decoder_lstm, decoder_dense = build_model(
    src_vocab_size, tgt_vocab_size, embedding_dim, lstm_units
)

model.fit(
    [source_tensor, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=300
)

model.save_weights('es_inv.weights.h5')
