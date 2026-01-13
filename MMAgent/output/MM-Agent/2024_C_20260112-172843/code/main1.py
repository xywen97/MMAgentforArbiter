# Here is the Python code.

import pandas as pd
import numpy as np

# Load the dataset
dataset_path = 'Wimbledon_featured_matches.csv'
data = pd.read_csv(dataset_path)

# Display initial data information
print("Initial Data Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Define a function to calculate momentum metrics
def calculate_momentum_metrics(df):
    metrics = {
        'points_won_p1': 0,
        'points_won_p2': 0,
        'unforced_errors_p1': 0,
        'unforced_errors_p2': 0,
        'aces_p1': 0,
        'aces_p2': 0,
        'double_faults_p1': 0,
        'double_faults_p2': 0,
        'serve_success_rate_p1': 0,
        'serve_success_rate_p2': 0,
        'critical_points_won_p1': 0,
        'critical_points_won_p2': 0
    }
    
    # Calculate metrics from the dataset
    for index, row in df.iterrows():
        # Points won
        metrics['points_won_p1'] += row['point_victor'] == 1
        metrics['points_won_p2'] += row['point_victor'] == 2
        
        # Unforced errors
        metrics['unforced_errors_p1'] += row['p1_unf_err']
        metrics['unforced_errors_p2'] += row['p2_unf_err']
        
        # Aces
        metrics['aces_p1'] += row['p1_ace']
        metrics['aces_p2'] += row['p2_ace']
        
        # Double faults
        metrics['double_faults_p1'] += row['p1_double_fault']
        metrics['double_faults_p2'] += row['p2_double_fault']
        
        # Serving success rate calculation
        if row['server'] == 1:  # Player 1 is serving
            if row['serve_no'] == 1:  # First serve
                if row['point_victor'] == 1:
                    metrics['serve_success_rate_p1'] += 1  # Successful first serve
            else:  # Second serve
                if row['point_victor'] == 1:
                    metrics['serve_success_rate_p1'] += 1  # Successful second serve
        else:  # Player 2 is serving
            if row['serve_no'] == 1:  # First serve
                if row['point_victor'] == 2:
                    metrics['serve_success_rate_p2'] += 1  # Successful first serve
            else:  # Second serve
                if row['point_victor'] == 2:
                    metrics['serve_success_rate_p2'] += 1  # Successful second serve
        
        # Critical points (e.g., break points and set points)
        if row['p1_break_pt'] == 1 and row['point_victor'] == 1:
            metrics['critical_points_won_p1'] += 1
        if row['p2_break_pt'] == 1 and row['point_victor'] == 2:
            metrics['critical_points_won_p2'] += 1
    
    # Calculate serve success rates
    total_serves_p1 = df[df['server'] == 1].shape[0]
    total_serves_p2 = df[df['server'] == 2].shape[0]
    
    metrics['serve_success_rate_p1'] = metrics['serve_success_rate_p1'] / total_serves_p1 if total_serves_p1 > 0 else 0
    metrics['serve_success_rate_p2'] = metrics['serve_success_rate_p2'] / total_serves_p2 if total_serves_p2 > 0 else 0

    return metrics

# Calculate momentum metrics
momentum_metrics = calculate_momentum_metrics(data)

# Display calculated metrics
print("\nCalculated Momentum Metrics:")
for key, value in momentum_metrics.items():
    print(f"{key}: {value}")

# Save metrics to a CSV file
metrics_df = pd.DataFrame([momentum_metrics])
metrics_df.to_csv('momentum_metrics.csv', index=False)

print("\nMomentum metrics saved to 'momentum_metrics.csv'")