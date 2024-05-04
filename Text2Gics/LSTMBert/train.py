"""
执行训练
"""
from LSTMBert.utils import gics_loss, gics_acc
from tensorflow.keras.optimizers import *
import warnings
from LSTMBert.structure import *

from LSTMBert.model import IndustryClassifierModel

warnings.filterwarnings('ignore')

for task in ['r2']:

    IndustryClassifierModel(locals()[task]).train(
        save_path=task,
        optimizer=Nadam(),
        loss=gics_loss,
        metrics=gics_acc,
        batch_size=8,
        start=42000,
        exchange_frequency=5,
        batches=1000,
        save_period=1000,
        eval_period=50,
        valid_batches=5,
    )
