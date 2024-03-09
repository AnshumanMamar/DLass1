import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist 
import wandb

wandb login a2qs3wd4ef5rg6th7yj8uxw3e4crvbt6

wandb.inti(Project="DL-ASS1", name="question1s")

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

# Define class names for reference
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create a grid for plotting
plt.figure(figsize=(10, 10))
for i in range(5,0,-1):
    j=1
# Loop through each classsou
for i in range(10):
    # Find the index of the first occurrence of the class in the labels
    index = next(j for j, label in enumerate(y_train) if label == i)
    
    # Plot the image
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

# Show the plot
plt.show()

