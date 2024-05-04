from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from config import y1, y2, y3
from settings import LOSS_WEIGHTS
from tensorflow.keras import backend as K

__all__ = ['gics_loss', 'gics_acc']

w1, w2, w3, w4 = LOSS_WEIGHTS

def gics_loss(y_true, y_pred):
    g1_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y1])
    g1_pred = K.concatenate([K.max(y_pred[:, d], axis=-1, keepdims=True) for d in y1])
    loss_1 = categorical_crossentropy(g1_true, K.softmax(g1_pred))

    g2_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y2])
    g2_pred = K.concatenate([K.max(y_pred[:, d], axis=-1, keepdims=True) for d in y2])
    loss_2 = categorical_crossentropy(g2_true, K.softmax(g2_pred))

    g3_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y3])
    g3_pred = K.concatenate([K.max(y_pred[:, d], axis=-1, keepdims=True) for d in y3])
    loss_3 = categorical_crossentropy(g3_true, K.softmax(g3_pred))

    loss_4 = categorical_crossentropy(y_true, K.softmax(y_pred))
    return w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4

def gics_vert_loss(y_true, y_pred):
    loss_1 = K.mean(K.square(y_pred[:, :1]))
    loss_2 = K.mean(K.square(y_pred[:, :2]))
    loss_3 = K.mean(K.square(y_pred[:, :3]))
    loss_4 = K.mean(K.square(y_pred[:, :4]))
    return w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4


def acc1(y_true, y_pred):
    group_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y1])
    group_pred = K.concatenate([K.max(y_pred[:, d], axis=1, keepdims=True) for d in y1])
    return categorical_accuracy(group_true, K.softmax(group_pred))

def acc2(y_true, y_pred):
    group_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y2])
    group_pred = K.concatenate([K.max(y_pred[:, d], axis=-1, keepdims=True) for d in y2])
    return categorical_accuracy(group_true, K.softmax(group_pred))

def acc3(y_true, y_pred):
    group_true = K.concatenate([K.max(y_true[:, d], axis=-1, keepdims=True) for d in y3])
    group_pred = K.concatenate([K.max(y_pred[:, d], axis=-1, keepdims=True) for d in y3])
    return categorical_accuracy(group_true, K.softmax(group_pred))

def acc4(y_true, y_pred):
    return categorical_accuracy(y_true, K.softmax(y_pred))

gics_acc = [acc1, acc2, acc3, acc4]
