import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class PlotGenerator:
    def generate_histograms(self, data):
        """
        Generate histograms for all columns in the dataframe.

        Args:
            data: The dataset to generate histograms for.

        """
        # Select only numeric columns
        numeric_columns = data.select_dtypes(include='number').columns

        if len(numeric_columns) == 0:
            return None

        # Number of rows and columns for subplots based on the number of numeric columns
        n_cols = 4  # you can set this to any number you'd like
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate number of rows

        # Creating subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

        # Flatten the axes array in case there are multiple rows
        axes = axes.flatten()

        # Plotting histograms for each numeric column
        for i, column in enumerate(numeric_columns):
            sns.histplot(data[column], kde=True, ax=axes[i], color='blue')
            axes[i].set_title(column)

        # Turn off axes for any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # save to EDA
        os.makedirs('EDA', exist_ok=True)
        plt.savefig('EDA/histograms.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return plt
    

    def generate_bar_charts(self, data):
        """
        Generate bar plots for categorical columns in the dataframe.

        Args:
            data: The dataset to generate bar plots for.

        """
        categorical_columns = []
        for column in data.columns:
            if (data[column].dtype == 'object' and data[column].nunique() < 20) or \
               (data[column].dtype == 'category' and data[column].nunique() < 20) or \
               (data[column].dtype == 'bool'):
                categorical_columns.append(column)

        if len(categorical_columns) == 0:
            return None

        # Number of rows and columns for subplots based on the number of categorical columns
        n_cols = 4  
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols  # Calculate number of rows

        # Creating subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

        # Flatten the axes array in case there are multiple rows
        axes = axes.flatten()

        # Plotting bar plots for each categorical column
        for i, column in enumerate(categorical_columns):
            sns.countplot(x=data[column], ax=axes[i])
            axes[i].set_title(column)
            axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

        # Turn off axes for any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Save to EDA
        os.makedirs('EDA', exist_ok=True)
        plt.savefig('EDA/bar_charts.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return plt
        

