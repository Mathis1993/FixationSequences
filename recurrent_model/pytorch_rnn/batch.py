from pytorch_deepgaze_based_lstm_batch import run_lstm
from pytorch_deepgaze_based_gru_batch import run_gru

hidden_sizes = [10, 30, 50]

for hidden_size in hidden_sizes:
    run_lstm(batch_size=8, lr=0.0001, n_epochs=15, gpu=True, hidden_size=hidden_size)
    run_gru(batch_size=8, lr=0.0001, n_epochs=15, gpu=True, hidden_size=hidden_size)