IMAGE_SIZE      = (299, 299)
MAX_VOCAB_SIZE  = 2000000
SEQ_LENGTH      = 25
EMBED_DIM       = 512
NUM_HEADS       = 6
FF_DIM          = 1024
SHUFFLE_DIM     = 512
BATCH_SIZE      = 64
EPOCHS          = 30

REDUCE_DATASET  = False
NUM_TRAIN_IMG   = 68363
NUM_VALID_IMG   = 20000
TRAIN_SET_AUG   = True
VALID_SET_AUG   = False
TEST_SET        = False

OBJECTIVE       = "4-4K"

if OBJECTIVE == "4-3E":
    train_data_json_path = "dataset/annotations/nia0403.train.eng.json"
    valid_data_json_path = "dataset/annotations/nia0403.valid.eng.json"
    text_data_json_path  = "dataset/annotations/nia0403.text.eng.json"
    SAVE                 = "model/4-3/english/"
elif OBJECTIVE == "4-3K":
    train_data_json_path = "dataset/annotations/nia0403.train.kor.json"
    valid_data_json_path = "dataset/annotations/nia0403.valid.kor.json"
    text_data_json_path  = "dataset/annotations/nia0403.text.kor.json"
    SAVE                 = "model/4-3/korean/"
elif OBJECTIVE == "4-4E":
    train_data_json_path = "dataset/annotations/nia0404.train.eng.json"
    valid_data_json_path = "dataset/annotations/nia0404.valid.eng.json"
    text_data_json_path  = "dataset/annotations/nia0404.text.eng.json"
    SAVE                 = "model/4-4/english/"
else:
    train_data_json_path = "dataset/annotations/nia0404.train.kor.json"
    valid_data_json_path = "dataset/annotations/nia0404.valid.kor.json"
    text_data_json_path  = "dataset/annotations/nia0404.text.kor.json"
    SAVE                 = "model/4-4/korean/"
