USE_GPU = True
MAX_LENGTH = {'brief': 5, 'main_business': 3, 'business_scope': 2}
LOSS_WEIGHTS = (1, 1, 1, 1)

BERT_LEN = 512
BERT_DIM = 768
HEAD_NUM = 12
FFD = 1024
TRAIN_PATH = 'data/trainset.csv'
VALID_PATH = 'data/validset.csv'
TEST_PATH = 'data/unlabel.csv'
