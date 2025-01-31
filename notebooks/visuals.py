import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

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

def plot_unemployment_rates(data, figsize=(12, 6)):
    """
    Create a bar plot of unemployment rates over time.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the unemployment data
    figsize (tuple): Figure size (width, height)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(data['Urtea'], data['Langabezia Tasa'], 
                color='#3498db', alpha=0.8)

    # Customize the plot
    ax.set_title('Langabezia', pad=20, fontsize=14)
    ax.set_xlabel('Urtea')
    ax.set_ylabel('Langabezi Tasa (%)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_unemployment_comparison(data, columns_to_plot, column_labels=None, figsize=(12, 6)):
    """
    Create a grouped bar plot comparing different unemployment rates over time.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the unemployment data
    columns_to_plot (list): List of column names to plot
    column_labels (list): Optional list of labels for the legend
    figsize (tuple): Figure size (width, height)
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bar positions
    years = data['Urtea'].values
    n_columns = len(columns_to_plot)
    width = 0.8 / n_columns  # Adjust bar width based on number of columns
    
    # Create color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    
    # Plot bars for each column
    for i, column in enumerate(columns_to_plot):
        # Calculate position for this set of bars
        positions = np.arange(len(years)) + i * width - (n_columns-1) * width/2
        
        # Create bars
        label = column_labels[i] if column_labels else column
        bars = ax.bar(positions, data[column], width,
                     label=label, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_title('Lagabezi tasa denboran zehar', pad=20, fontsize=14)
    ax.set_xlabel('Urtea')
    ax.set_ylabel('Langabezi Tasa (%)')
    
    # Set x-axis ticks at bar group centers
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    
    # Add gridlines and legend
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

def plot_unemployment_rates_pred(data, predict_years=3, figsize=(12, 6)):
    """
    Create a bar plot of unemployment rates over time with future predictions.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the unemployment data
    predict_years (int): Number of years to predict into the future
    figsize (tuple): Figure size (width, height)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for prediction
    X = data['Urtea'].values.reshape(-1, 1)
    y = data['Langabezia Tasa'].values
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future years and predictions
    last_year = data['Urtea'].max()
    future_years = np.array(range(last_year + 1, last_year + predict_years + 1))
    future_predictions = model.predict(future_years.reshape(-1, 1))
    
    # Create bars for actual data
    bars = ax.bar(data['Urtea'], data['Langabezia Tasa'], 
                 color='#3498db', alpha=0.8, label='Actual')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Plot prediction line for both historical and future data
    all_years = np.concatenate([X.flatten(), future_years])
    all_predictions = model.predict(all_years.reshape(-1, 1))
    
    # Plot the trend line for historical data
    ax.plot(X.flatten(), model.predict(X), 
            color='#e74c3c', linestyle='--', alpha=0.5)
    
    # Plot the prediction line and confidence interval
    ax.plot(future_years, future_predictions, 
            color='#e74c3c', linestyle='--', 
            label='Predicted', linewidth=2)
    
    # Add points for predictions
    ax.scatter(future_years, future_predictions, 
              color='#e74c3c', zorder=5)
    
    # Add labels for predicted values
    for year, pred in zip(future_years, future_predictions):
        ax.text(year, pred, f'{pred:.2f}%',
                ha='center', va='bottom')
    
    # Customize the plot
    ax.set_title('Unemployment Rate Over Time with Future Predictions', 
                pad=20, fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Unemployment Rate (%)')
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_unemployment_rates_poly(data, predict_years=3, degree=2, figsize=(12, 6)):
    """
    Create a bar plot of unemployment rates over time with future predictions using polynomial regression.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the unemployment data
    predict_years (int): Number of years to predict into the future
    degree (int): Degree of the polynomial regression
    figsize (tuple): Figure size (width, height)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for prediction
    X = data['Urtea'].values.reshape(-1, 1)
    y = data['Langabezia Tasa'].values
    
    # Create and fit the polynomial regression model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Generate future years and predictions
    last_year = data['Urtea'].max()
    future_years = np.array(range(last_year + 1, last_year + predict_years + 1))
    future_predictions = model.predict(future_years.reshape(-1, 1))
    
    # Create smooth curve points for plotting
    X_smooth = np.linspace(data['Urtea'].min(), future_years.max(), 300).reshape(-1, 1)
    y_smooth = model.predict(X_smooth)
    
    # Create bars for actual data
    bars = ax.bar(data['Urtea'], data['Langabezia Tasa'], 
                 color='#3498db', alpha=0.8, label='Actual')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Plot the smooth curve for historical and future predictions
    ax.plot(X_smooth, y_smooth, 
            color='#e74c3c', linestyle='--', 
            label=f'Polynomial Regression (degree={degree})', 
            alpha=0.8)
    
    # Add points and labels for predictions
    ax.scatter(future_years, future_predictions, 
              color='#e74c3c', zorder=5, label='Predicted Points')
    
    for year, pred in zip(future_years, future_predictions):
        # Ensure predictions don't go below 0
        pred_value = max(0, pred)
        ax.text(year, pred_value, f'{pred_value:.2f}%',
                ha='center', va='bottom')
    
    # Customize the plot
    ax.set_title(f'Unemployment Rate Over Time with Polynomial Predictions (degree={degree})', 
                pad=20, fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Unemployment Rate (%)')
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Ensure y-axis doesn't go below 0
    ax.set_ylim(bottom=0)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Return the model's R-squared score for the training data
    r2_score = model.score(X, y)
    
    return fig, ax, r2_score