import pandas as pd
import numpy as np
import logging

# Set up logging to track computation steps
logging.basicConfig(level=logging.INFO)

# The model class
class Model2:
    def __init__(self, data):
        self.data = data
        self.alpha = 1.0  # placeholder for parameter alpha
        self.beta = 1.0   # placeholder for parameter beta
        self.gamma = 1.0  # placeholder for parameter gamma

    def calculate_performance_scores(self):
        # Calculate points won, unforced errors, aces, double faults, serve success rate, etc.
        self.data['P_A'] = self.data['p1_points_won']  # Points won by Alcaraz
        self.data['P_J'] = self.data['p2_points_won']  # Points won by Djokovic

        # Calculate unforced errors
        self.data['UE_A'] = self.data['p1_unf_err']
        self.data['UE_J'] = self.data['p2_unf_err']

        # Calculate aces and double faults
        self.data['A_A'] = self.data['p1_ace']
        self.data['A_J'] = self.data['p2_ace']
        self.data['DF_A'] = self.data['p1_double_fault']
        self.data['DF_J'] = self.data['p2_double_fault']

        # Total serves for success rate
        self.data['T_A'] = self.data['p1_ace'] + self.data['p1_double_fault'] + self.data['p1_games']  # Total serves for Alcaraz
        self.data['T_J'] = self.data['p2_ace'] + self.data['p2_double_fault'] + self.data['p2_games']  # Total serves for Djokovic

        # Calculate Serve Success Rate
        self.data['SSR_A'] = self.data['A_A'] / self.data['T_A']
        self.data['SSR_J'] = self.data['A_J'] / self.data['T_J']

        # Calculate Performance Scores
        self.data['P_A'] = (self.data['P_A'] - self.data['UE_A'] + self.data['A_A'] - self.data['DF_A'] + self.data['SSR_A']).fillna(0)
        self.data['P_J'] = (self.data['P_J'] - self.data['UE_J'] + self.data['A_J'] - self.data['DF_J'] + self.data['SSR_J']).fillna(0)

        logging.info("Performance scores calculated.")

    def calculate_momentum_index(self):
        # Calculate Psychological State based on a simple formula (this can be more complex)
        self.data['P_sy_A'] = np.where(self.data['P_A'] > self.data['P_J'], 1, 0)  # simplistic psychological state
        self.data['P_sy_J'] = np.where(self.data['P_A'] < self.data['P_J'], 1, 0)

        # Momentum Index calculation
        self.data['M_A'] = self.alpha * self.data['P_A'] - self.beta * self.data['P_J'] + self.gamma * self.data['P_sy_A']
        self.data['M_J'] = self.alpha * self.data['P_J'] - self.beta * self.data['P_A'] + self.gamma * self.data['P_sy_J']

        logging.info("Momentum index calculated.")

    def run_analysis(self):
        self.calculate_performance_scores()
        self.calculate_momentum_index()

# The function to complete the current Task
def task2():
    # Load the dataset
    logging.info("Loading dataset...")
    data = pd.read_csv('Wimbledon_featured_matches.csv')

    # Create a model instance
    model = Model2(data)

    # Run the analysis
    model.run_analysis()

    # Save the results
    logging.info("Saving results...")
    model.data.to_csv('momentum_metrics.csv', index=False)

    logging.info("Task completed. Results saved to 'momentum_metrics.csv'.")

if __name__ == '__main__':
    # complete task 
    task2()