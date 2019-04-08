import pandas as pd
#create empty df to store the results and store it to disk
results = pd.DataFrame(columns = ["batch_size", "n_epochs", "learning_rate", "mean_accuracy_per_image", "mean_test_loss", "mean_validation_loss", "mean_train_loss", "number_of_hits", "number_of_test_images", "number_of_fixations"])
results.to_csv("results.csv", index=False, header=True)