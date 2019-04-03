import pandas as pd
#create empty df to store the results and store it to disk
results = pd.DataFrame(columns = ["batch_size", "learning_rate", "mean_accuracy_per_image", "mean_test_loss", "mean_validation_loss", "mean_train_loss"])
results.to_csv("results.csv", index=False, header=True)