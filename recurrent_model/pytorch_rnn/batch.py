from pytorch_deepgaze_based_lstm_batch import run_lstm
from pytorch_deepgaze_based_gru_batch import run_gru

lrs = [0.00005, 0.00001]

for lr in lrs:
    run_lstm(lr, 40, True)
    run_gru(lr, 40, True)