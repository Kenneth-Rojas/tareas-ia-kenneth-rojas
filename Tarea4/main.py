# main.py

modo = input("Modo de traducción ('es' para español→inventado, 'inv' para inventado→español): ").strip()

if modo == 'es':
    from data import src_word2idx, tgt_idx2word, max_len_src
    from inference import encode_input, decode_sequence
elif modo == 'inv':
    from data_invertido import src_word2idx, tgt_idx2word, max_len_src
    from inference_invertido import encode_input, decode_sequence
else:
    print("Modo inválido. Usa 'es' o 'inv'.")
    exit()

test_sentence = input("Escribe la frase a traducir: ").strip()

input_seq = encode_input(test_sentence, src_word2idx, max_len_src)
translation = decode_sequence(input_seq)

print("\nFrase original:", test_sentence)
print("Traducción:", translation)
