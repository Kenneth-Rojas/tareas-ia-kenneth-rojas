# main.py
from data import src_word2idx, tgt_idx2word, max_len_src
from inference import encode_input, decode_sequence

# Oración de prueba
test_sentence = "nos vemos mañana"

# Codificar la oración de entrada
input_seq = encode_input(test_sentence, src_word2idx, max_len_src)

# Obtener traducción usando decode_sequence que solo toma input_seq
translation = decode_sequence(input_seq)

print("Frase:", test_sentence)
print("Traducción:", translation)
