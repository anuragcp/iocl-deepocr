MAL_VECTOR = 'ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹാിീുൂൃെേൈൊോൌ്ൎൗൺൻർൽൾ.,'

ASCII_VECTOR = '-+=!@#$%^&*(){}[]|\'"\\/?<>;:0123456789'

ENG_VECTOR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

CHAR_VECTOR = ENG_VECTOR

letters = [letter for letter in CHAR_VECTOR] # letter array

num_classes = len(letters) + 1               # total length of output chars + CTC separation char

img_w, img_h = 160, 32

# Network parameters
batch_size = 32
val_batch_size = 16

downsample_factor = 4
max_text_len = 20                            # maximum text length output
