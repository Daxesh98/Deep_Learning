import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Plot the learning rate versus the loss
def semi_log(history):
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])  # we want the x-axis (learning rate) to be log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")
    plt.show()

# Plot the loss curves
def loss_curve(history1):
    pd.DataFrame(history1.history).plot()
    plt.title("Model_1 training curves")
    plt.show()