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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))

        # Flatten the axes array in case there are multiple rows
        axes = axes.flatten()

        # Plotting histograms for each numeric column
        for i, column in enumerate(numeric_columns):
            sns.histplot(data[column], kde=True, ax=axes[i], color='blue', bins=20)
            axes[i].set_title(column, fontsize=16)  
            axes[i].tick_params(axis='both', labelsize=14)  
            axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=14)  
            axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=14)

        # Turn off axes for any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # save to EDA
        os.makedirs('Results/EDA', exist_ok=True)
        plt.savefig('Results/EDA/histograms.png', bbox_inches='tight', pad_inches=0)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.8, n_rows * 5.8))

        # Flatten the axes array in case there are multiple rows
        axes = axes.flatten()

        # Plotting bar plots for each categorical column
        for i, column in enumerate(categorical_columns):
            sns.countplot(x=data[column], ax=axes[i], color='#BD1052')  
            axes[i].set_title(column, fontsize=16)  
            axes[i].tick_params(axis='x', rotation=45, labelsize=14)  
            axes[i].tick_params(axis='y', labelsize=14)  
            axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=14)  
            axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=14)  

        # Turn off axes for any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        # Save to EDA
        os.makedirs('Results/EDA', exist_ok=True)
        plt.savefig('Results/EDA/bar_charts.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return plt
    
    def generate_box_plots_metrics(self, metrics):
        # Create a figure with 3 subplots (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Create boxplot for 'accuracy' on the first subplot
        sns.boxplot(x='model', y='accuracy', data=metrics, color='#0A6CF1', linewidth=2, fliersize=6, ax=axes[0])
        sns.stripplot(x='model', y='accuracy', data=metrics, color='#02275A', jitter=True, size=7, alpha=1, ax=axes[0])
        axes[0].set_title('Accuracy Comparison', fontsize=16)
        axes[0].set_xlabel('Model', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)

        # Create boxplot for 'f1' on the second subplot
        sns.boxplot(x='model', y='f1', data=metrics, color='#0A6CF1', linewidth=2, fliersize=6, ax=axes[1])
        sns.stripplot(x='model', y='f1', data=metrics, color='#02275A', jitter=True, size=7, alpha=1, ax=axes[1])
        axes[1].set_title('F1 Score Comparison', fontsize=16)
        axes[1].set_xlabel('Model', fontsize=14)
        axes[1].set_ylabel('F1 Score', fontsize=14)

        # Create boxplot for 'roc_auc' on the third subplot
        sns.boxplot(x='model', y='roc_auc', data=metrics, color='#0A6CF1', linewidth=2, fliersize=6, ax=axes[2])
        sns.stripplot(x='model', y='roc_auc', data=metrics, color='#02275A', jitter=True, size=7, alpha=1, ax=axes[2])
        axes[2].set_title('ROC AUC Comparison', fontsize=16)
        axes[2].set_xlabel('Model', fontsize=14)
        axes[2].set_ylabel('ROC AUC', fontsize=14)

        # Adjust layout for better spacing between plots
        plt.tight_layout()

        # Save to Results
        os.makedirs('Results/ModelOptimization', exist_ok=True)
        plt.savefig('Results/ModelOptimization/box_plots_metrics.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        return plt

        

