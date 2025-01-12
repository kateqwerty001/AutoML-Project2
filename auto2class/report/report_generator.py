from pylatex import Command, Document, Section, Subsection, Tabular
from pylatex.utils import NoEscape
from pylatex.table import Table 
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pylatex import Section, Subsection, Figure, NoEscape, Subsubsection
import seaborn as sns
import matplotlib.pyplot as plt
from .plot_generator import PlotGenerator

class ReportGenerator:
    def __init__(self, dataset, dataset_name, optimizer):
        self.doc = Document()
        self.dataset = pd.DataFrame(dataset)
        self.dataset_name = dataset_name
        self.optimizer = optimizer

        # delete clf__ from the column names
        self.optimizer.params_rf.columns = [col.replace('clf__', '') for col in self.optimizer.params_rf.columns]
        self.optimizer.params_dt.columns = [col.replace('clf__', '') for col in self.optimizer.params_dt.columns]
        self.optimizer.params_xgb.columns = [col.replace('clf__', '') for col in self.optimizer.params_xgb.columns]

    def make_small_margins(self):
        '''
        This method reduces the margins of the document to make it more compact
        '''
        self.doc.packages.append(Command('usepackage', 'geometry'))
        self.doc.packages.append(Command('geometry', 'margin=0.5in'))


    def new_page(self):
        '''
        This method adds a new page to the report
        '''
        self.doc.append(NoEscape(r'\newpage'))


    def add_title(self):
        '''
        This method adds a title to the report
        '''
        self.doc.preamble.append(Command('title', "Report on " + self.dataset_name + " dataset"))
        self.doc.preamble.append(Command('author', 'Auto2Class'))
        self.doc.append(NoEscape(r'\maketitle'))
        self.new_page()   


    def add_table_of_contents(self):
        '''
        This method adds a table of contents to the report
        '''
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.new_page()


    def add_info_table(self):
        '''
        This method adds a table with the dataset information (using .info())
        '''
        # Capture dataset.info() into a buffer
        buffer = io.StringIO()
        self.dataset.info(buf=buffer)
        dataset_info = buffer.getvalue()

        # Extract parts of dataset info() for table formatting
        lines = dataset_info.split('\n')
        table_rows = []

        # Process lines to extract table header and rows
        for line in lines:
            if "Non-Null Count" in line and "Dtype" in line:
                header = ["Column", "Non-Null Count", "Dtype"]
            elif line.strip() and line.strip()[0].isdigit():
                parts = line.split()
                column_name = parts[1]
                non_null_count = parts[2]
                dtype = parts[-1]
                table_rows.append([column_name, non_null_count, dtype])

        # Convert extracted information to a dataframe
        info_df = pd.DataFrame(table_rows, columns=header)

        # Use the print_dataframe method to print the dataframe
        self.print_dataframe(
            df=info_df,
            caption='Dataset Columns Information',
            num_after_dot=0
        )

    def add_describe_info(self):
        '''
        This method adds a table with the dataset's descriptive statistics (using .describe())
        '''
        # Calculate descriptive statistics and reset the index to include row labels
        describe_data = self.dataset.describe().transpose().reset_index()
        describe_data.rename(columns={'index': 'Column Name/Statistic'}, inplace=True)

        # Use the print_dataframe method to print the dataframe
        self.print_dataframe(
            df=describe_data,
            caption='Dataset Descriptive Statistics',
            num_after_dot=2
        )

    def add_bar_charts(self):
        '''
        This method adds bar charts for each categorical column in the dataset
        '''
        self.new_page()

        plt = PlotGenerator().generate_bar_charts(self.dataset)
        if plt is None:
            return
        
        with self.doc.create(Subsubsection('Bar Charts of Categorical columns')):
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image('EDA/bar_charts.png', width='460px')
                fig.add_caption('Bar Charts of Categorical columns')


    def add_histograms(self):
        '''
        This method adds histograms for each numerical column in the dataset
        '''
        plt = PlotGenerator().generate_histograms(self.dataset)
        if plt is None:
            return
        
        with self.doc.create(Subsubsection('Histograms of Numerical columns')):
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image('EDA/histograms.png', width='460px')
                fig.add_caption('Histograms of Numerical columns')
        return 
    
    
    def print_dataframe(self, df, caption, num_after_dot=2, no_index=False):
        '''
        This method prints the entire dataframe to the report.

        Args:
            df: The dataframe to be printed in the report.
            caption: The caption for the table in the report.
            num_after_dot: The number of decimal places to round numerical values in the dataframe.
        '''
        # Create the table with a caption
        with self.doc.create(Table(position='h!')) as table:
            table.add_caption(caption)

            self.doc.append(NoEscape(r'\vspace{0.2cm}')) # Add vertical space
            self.doc.append(NoEscape(r'\centering'))  # Center the table

            # Extract header and rows from the dataframe
            header = list(df.columns)
            table_rows = df.values.tolist()

            # Round numerical values to the specified number of decimal places
            rounded_rows = []
            for row in table_rows:
                rounded_row = [
                    round(value, num_after_dot) if isinstance(value, (float, int)) else value
                    for value in row
                ]
                rounded_rows.append(rounded_row)

            # Adjust the number of columns to match the dataframe
            with table.create(Tabular('|c|' * (len(header) + 1))) as tabular:
                tabular.add_hline()
                if no_index==False:
                    tabular.add_row(['Index'] + header)  # Add index column
                tabular.add_hline()

                # Add all rows to the table
                for i, row in enumerate(rounded_rows):
                    tabular.add_row([i] + row)

                tabular.add_hline()
                        

    def generate_report(self):
        '''
        This method generates the report
        '''
        self.make_small_margins()  # Reduce the margins of the document
        self.add_title()  # Add the title to the report
        self.add_table_of_contents()  # Add table of contents

        with self.doc.create(Section('Exploratory Data Analysis')):

            with self.doc.create(Subsection('Non-Null Count, Dtype of features')):
                self.add_info_table()  # Add dataset info() table

            with self.doc.create(Subsection('Descriptive Statistics')):
                self.add_describe_info()  # Add dataset describe() table
            
            self.new_page()
            with self.doc.create(Subsection('Distribution of features')):
                self.add_histograms() # Add histograms for numerical columns
                self.add_bar_charts() # Add bar charts for categorical columns

        self.new_page()
        with self.doc.create(Section('Model Optimization Results')):
            self.print_dataframe(self.optimizer.params_rf.transpose().reset_index().rename(columns={"index": "Metric/Hyperp.\ Iteration"}), 'Random Forest Hyperparameters and achivied metrics', num_after_dot=4)
            self.print_dataframe(self.optimizer.params_dt.transpose().reset_index().rename(columns={"index": "Metric/Hyperp. \ Iteration"}), 'Decision Tree Hyperparameters and achivied metrics', num_after_dot=4)
            self.print_dataframe(self.optimizer.params_xgb.transpose().reset_index().rename(columns={"index": "Metric/Hyperp. \ Iteration"}), 'XGBoost Hyperparameters and achivied metrics', num_after_dot=4)
        
        self.doc.generate_pdf('report', clean_tex=False)

    

