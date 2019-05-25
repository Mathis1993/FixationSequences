from pytorch_deepgaze_based_lstm_batch import run_lstm
from pytorch_deepgaze_based_gru_batch import run_gru

lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

for lr in lrs:
    run_lstm(lr, 10, True)
    run_gru(lr, 10, True)