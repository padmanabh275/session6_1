import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import glob
import os

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
        
        # Create plots directory if it doesn't exist
        Path('plots').mkdir(exist_ok=True)
        
        # Clean up old plot files
        for plot_file in glob.glob('plots/*.png'):
            os.remove(plot_file)
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_metrics(self):
        # Use classic matplotlib style
        plt.style.use('classic')
        
        # Create a figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics Overview', fontsize=16)
        
        # Plot 1: Loss Curves
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Accuracy Curves
        ax2.plot(self.epochs, self.train_accuracies, 'g-', label='Training Accuracy')
        ax2.plot(self.epochs, self.val_accuracies, 'orange', label='Validation Accuracy')
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Learning Rate Schedule
        ax3.plot(self.epochs, self.learning_rates, 'purple')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Plot 4: Training vs Validation Accuracy Distribution
        ax4.boxplot([self.train_accuracies, self.val_accuracies], 
                   labels=['Training', 'Validation'])
        ax4.set_title('Accuracy Distribution')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('plots/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, np.abs(np.array(self.train_accuracies) - np.array(self.val_accuracies)), 
                'r-', label='|Train-Val| Accuracy Gap')
        plt.title('Model Convergence Analysis')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy Gap (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/convergence.png', dpi=300, bbox_inches='tight')
        plt.close() 