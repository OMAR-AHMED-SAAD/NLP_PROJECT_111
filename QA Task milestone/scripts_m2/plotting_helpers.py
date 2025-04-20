from typing import List

import matplotlib.pyplot as plt

def plot_loss_curves(train_loss: List[float], val_loss: List[float]):
    """
    Plots the training and validation loss curves.

    Args:
        train_loss: A list of training loss values per epoch/step.
        val_loss: A list of validation loss values per epoch/step.
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example Usage:
    # Replace with your actual loss data
    example_train_loss = [2.5, 1.8, 1.2, 0.9, 0.7]
    example_val_loss = [2.8, 2.1, 1.5, 1.3, 1.1]

    plot_loss_curves(example_train_loss, example_val_loss)
