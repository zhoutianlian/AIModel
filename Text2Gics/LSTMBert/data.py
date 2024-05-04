from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import codecs
from keras_bert import Tokenizer
from config import gics4, VOCAB_PATH
from glob import glob
import re
from settings import *


__all__ = ['data_generator', 'pre_treat', 'join', 'get_token', 'TOKEN_NUM', 'TEXT_COL']
TEXT_COL = ['company_name', 'brief', 'main_business', 'business_scope']
CAT_NUM = len(gics4)
token_dict = {}

with codecs.open(VOCAB_PATH, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)
TOKEN_NUM = len(token_dict)

def pre_treat(text):
    """
    这个方法用于清洗文本，使其更加规范
    :param text:
    :return:
    """
    text = text.replace('\n', '')
    # 替换所有缩率词
    text = re.sub(r'([A-Z][a-z]*)\.', r'\1', text)
    text = text.replace(', Inc', ' Inc')

    # 替换所有编号
    text = re.sub(r'([0-9]+)\.([^0-9])', r'(\1) \2', text)

    # 替换英文逗号为中文逗号并去掉数位分隔符
    text = re.sub(r'([0-9]+),([0-9]+)', r'\1\2', text)
    text = text.replace(',', '，')

    # 替换断句符号并保留小数点
    text = re.sub(r'([0-9]+)\.+([0-9]+)', r'\1*\2', text)
    for s in '，；：！？…,;:.!?':
        text = text.replace(s, '。')
    text = text.replace('*', '.')

    # 替换引用并保留缩略符号
    text = re.sub(r'([a-zA-Z0-9])\'', r'\1`', text)
    for s in "“”‘’`'":
        text = text.replace(s, '"')
    text = text.replace("`", "'")

    # 替换短划线为破折号并保留负号和化学表达式
    text = re.sub(r'-([0-9]+)', r'=\1', text)
    text = re.sub(r'([0-9]+)-', r'=\1', text)
    text = text.replace('-', '—').replace('=', '-')

    # 消除非英文空格
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1_\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1_\2', text)  # 重复一次避免错过重叠部分的匹配
    text = text.replace(' ', '').replace('_', ' ')

    # 替换注释符号
    for left, right in zip('【{[（<「', '】}]）>」'):
        text = text.replace(left, '(')
        text = text.replace(right, ')')

    # 排除多重断句
    if '。。' in text:
        text = text.replace('。。', '。')

    if not text.endswith('。'):
        text = text + '。'

    return text


def join(t):
    t = ''.join(t)
    while '。。' in t:
        t = t.replace('。。', '。')
    return t


def get_token():
    def to_token(text):
        tok, seg = tokenizer.encode(first=text)
        return np.array(tok), np.array(seg)
    return to_token


def random_exchanger(dict_path: str, target: str = None, prob: float = 0.5, count: int = -1):
    """
    将词典中词汇替换为指定词或随机同类词
    :param dict_path:
    :param target:
    :param prob: 替换的概率
    :param count: 替换多少个 -1表示全部替换
    :return:
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')

    while '' in words:
        words.remove('')

    if target is not None:
        def exchanger(text):
            for w in words:
                text = text.replace(w, target)
            return text
    else:
        def exchanger(text):
            np.random.shuffle(words)
            words_count = len(words)
            for i in range(len(words)):
                if words[i] in text and np.random.rand() < prob:
                    text = text.replace(words[i], words[np.random.randint(0, words_count)], count)
            return text

    return exchanger


def exchange_generator(data, exchange_frequency, max_length):
    """
    可以交换句子的生成器
    :param data:
    :param exchange_frequency:
    :param max_length:
    :return:
    """
    df = None

    ex_dicts = list()
    ex_num = 0
    for ed_path in glob('exchange/*.txt'):
        # name = re.findall('([a-zA-Z]+)\.txt', ed_path)[0]
        ex_dicts.append(random_exchanger(ed_path))
        ex_num += 1

    data['gics_id'] = data['gics_code'].apply(lambda g: np.where(gics4==g)[0][0])

    while True:
        batch_size = yield df
        # 当batch_size为None时，返回所有数据
        if batch_size is None:
            df = data[['gics_id', 'company_name']]
            df[TEXT_COL[1:]] = data[TEXT_COL[1:]].applymap(lambda t: t.split('。'))

            for c, ml in max_length.items():
                df.loc[:, c] = df[c].apply(lambda t: t[: min([len(t), ml])])

            df[TEXT_COL[1:]] = df[TEXT_COL[1:]].applymap(lambda t: '。'.join(t))
            df['text'] = df[TEXT_COL].apply(join, axis=1)

            for ed in ex_dicts:
                df['text'] = df['text'].apply(ed)  # 对等价名词随机替换

            df = df[['gics_id', 'text']]

        # 否则，返回随机变换过的一个batch
        else:
            df = list()
            # 随机生成一个batch长度的行业id，并遍历
            for i in np.random.randint(0, CAT_NUM, batch_size):
                industry_data = data.query('gics_id == @i')
                # 随机提取同行业的3-10行数据
                lucky = industry_data.iloc[np.random.randint(0, len(industry_data), np.random.randint(3, 10))][TEXT_COL]
                # 对长文本进行抽取
                lucky[TEXT_COL[1:]] = lucky[TEXT_COL[1:]].applymap(lambda t: t.split('。'))

                # 限制最大长度
                for c, ml in max_length.items():
                    lucky.loc[:, c] = lucky[c].apply(lambda t: t[: min([len(t), ml])])

                # 随机次数不超过exchange_frequency 随机字段
                for col in np.random.randint(1, len(TEXT_COL), np.random.randint(0, exchange_frequency)):
                    host = lucky.iloc[0, col]  # 提取主句
                    j = np.random.randint(0, len(host))  # 主句第几句话
                    row = np.random.randint(1, len(lucky))  # 客句取第几行数据
                    cust = lucky.iloc[row, col]  # 提取客句
                    k = np.random.randint(0, len(cust))  # 客句第几句话
                    host[j] = cust[k]  # 置换
                    lucky.iloc[0, col] = host  # 填入

                joint_text = lucky.iloc[0]
                joint_text[TEXT_COL[1:]] = joint_text[TEXT_COL[1:]].apply(lambda t: '。'.join(t))
                joint_text = join(joint_text)
                df.append({'gics_id': i, 'text': joint_text})

            df = pd.DataFrame(df, index=range(batch_size))
            # 进行等价名词随机变换
            for ed in ex_dicts:
                df['text'] = df['text'].apply(ed)


def data_generator(data_path, exchange_frequency: int = 0,
                   return_text: bool = False):
    """
    数据生成器
    :param data_path:
    :param exchange_frequency: 同一行业下语句的交换率
    :param return_text: 是否返回文本
    :return:
    """
    data = pd.read_csv(data_path, dtype={'gics_code': int})
    data = data.fillna('')
    data[TEXT_COL] = data[TEXT_COL].applymap(pre_treat)

    for k in TEXT_COL[1:]:
        MAX_LENGTH.setdefault(k, np.inf)

    batch_x = None
    batch_y = None
    df = None

    generator = exchange_generator(data, exchange_frequency, MAX_LENGTH)
    next(generator)
    print('Using random dataset!')

    while not return_text:
        batch_size = yield batch_x, batch_y
        df = generator.send(batch_size)
        gics_id = df['gics_id'].values
        batch_y = to_categorical(gics_id, CAT_NUM, int)

        tok, seg = zip(*df['text'].apply(get_token()))
        batch_x = [pad_sequences(tok, BERT_LEN, padding='post'),
                   pad_sequences(seg, BERT_LEN, padding='post')]

    while return_text:
        batch_size = yield df
        df = generator.send(batch_size)


if __name__ == '__main__':
    dg = data_generator('data/trainset.csv', exchange_frequency=5)

    next(dg)
    y, *x = dg.send(32)
    print(y)
    print(x[0])
