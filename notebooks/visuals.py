import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_pos_neg_bars(data, categories, title='Positive and Negative Values', figsize=(12, 6)):
    """
    Create a bar plot with different colors for positive and negative values.
    
    Parameters:
    data (pd.Series): Series containing the values to plot
    categories (list-like): List of category labels for x-axis
    title (str): Title for the plot
    figsize (tuple): Figure size (width, height)
    """
    # Input validation
    if len(data) != len(categories):
        raise ValueError("Length of data and categories must match")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get indices for positive and negative values
    pos_mask = data >= 0
    neg_mask = data < 0
    
    # Create x positions for bars
    x = np.arange(len(data))
    
    # Plot positive values in green
    ax.bar(x[pos_mask], data[pos_mask], color='#2ecc71', alpha=0.8)
    
    # Plot negative values in grey
    ax.bar(x[neg_mask], data[neg_mask], color='#95a5a6', alpha=0.8)
    
    # Customize the plot
    ax.set_title(title, pad=20, fontsize=12)
    ax.set_ylabel('Value')
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

def plot_offers_comparison(df, offer_column, score1_column, score2_column,
                         label1="Store 1", label2="Store 2",
                         title="Offer Comparison",
                         figsize=(10, 10)):
    """
    Create a spider/radar plot comparing two sets of offers from DataFrame columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    offer_column (str): Name of column containing offers names
    score1_column (str): Name of column containing first set of scores
    score2_column (str): Name of column containing second set of scores
    label1 (str): Label for first dataset
    label2 (str): Label for second dataset
    title (str): Plot title
    figsize (tuple): Figure size
    """
    # Extract data from DataFrame
    offers = df[offer_column].tolist()
    dataset1 = df[score1_column].tolist()
    dataset2 = df[score2_column].tolist()
    
    # Number of variables
    num_vars = len(offers)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Add the first dataset
    values1 = dataset1 + [dataset1[0]]  # Complete the circle
    ax.plot(angles, values1, 'o-', linewidth=2, label=label1, color='#2ecc71')
    ax.fill(angles, values1, alpha=0.25, color='#2ecc71')
    
    # Add the second dataset
    values2 = dataset2 + [dataset2[0]]  # Complete the circle
    ax.plot(angles, values2, 'o-', linewidth=2, label=label2, color='#3498db')
    ax.fill(angles, values2, alpha=0.25, color='#3498db')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(offers)
    
    # Add title and legend
    plt.title(title, y=1.05, fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set chart limits based on data range
    max_val = max(max(dataset1), max(dataset2))
    plt.ylim(0, max_val * 1.1)  # Add 10% padding
    
    return fig, ax
