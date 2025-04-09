import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(csv_file, output_path="loss_plot.png"):
    data = pd.read_csv(csv_file)
    epochs = data["epoch"].dropna().astype(int)
    losses = data["loss"].dropna().astype(float)
    
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_accuracy_comparison(json_files, output_path="accuracy_comparison.png"):
    accuracies = []
    labels = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            accuracies.append(data["accuracy"])
            labels.append(f"patch={data['params']['patch_size']}, layers={data['params']['layers']}")
    
    plt.bar(labels, accuracies)
    plt.xlabel("Experiment")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy Comparison Across Experiments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()