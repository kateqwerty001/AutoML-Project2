from pylatex import Command, Document, Section, Subsection, Tabular
from pylatex.utils import NoEscape
from pylatex.table import Table 
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pylatex import Section, Subsection, Figure, NoEscape

class ReportGenerator:
    def __init__(self, dataset, dataset_name):
        self.doc = Document()
        self.dataset = pd.DataFrame(dataset)
        self.dataset_name = dataset_name


    def add_title(self):
        '''
        This method adds a title to the report
        '''
        self.doc.preamble.append(Command('title', "Report created by Auto2Class on '" + self.dataset_name + "' dataset"))
        self.doc.append(NoEscape(r'\maketitle'))


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

        with self.doc.create(Section('Exploratory Data Analysis Part')):
            with self.doc.create(Subsection('DataTypes and Non-Null Count')):
                self.doc.append(NoEscape(r'Information is in table 1'))
                # Create the table with a caption 
                with self.doc.create(Table(position='h!')) as table:
                    table.add_caption('Dataset Columns Information')

                    self.doc.append(NoEscape(r'\vspace{0.2cm}'))

                    self.doc.append(NoEscape(r'\centering')) # center the table

                    with table.create(Tabular('|c|c|c|')) as tabular:
                        tabular.add_hline()
                        tabular.add_row(header)
                        tabular.add_hline()
                        for row in table_rows:
                            tabular.add_row(row)
                        tabular.add_hline()

    def add_describe_info(self):
            '''
            This method adds a table with the dataset's descriptive statistics (using .describe())
            '''
            # Capture dataset.describe() into a buffer
            describe_data = self.dataset.describe().transpose()

            # Extract header and rows for descriptive statistics
            header = list(describe_data.columns)
            table_rows = describe_data.values.tolist()

            with self.doc.create(Subsection('Descriptive Statistics')):
                self.doc.append(NoEscape(r'Information is in table 2'))
                # Create the table with a caption 
                with self.doc.create(Table(position='h!')) as table:
                    table.add_caption('Dataset Descriptive Statistics')

                    self.doc.append(NoEscape(r'\vspace{0.2cm}'))

                    self.doc.append(NoEscape(r'\centering'))  # center the table

                    # Adjust the number of columns to match the data (1 column for "Statistic" and columns for each statistic)
                    with table.create(Tabular('|c|' + 'c|' * len(header))) as tabular:
                        tabular.add_hline()
                        tabular.add_row(['Statistic'] + header)  # Add a 'Statistic' column
                        tabular.add_hline()
                        for i, row in enumerate(table_rows):
                            # Round float values to 2 decimal places if they are floats
                            row = [round(x, 2) if isinstance(x, float) else x for x in row]
                            tabular.add_row([describe_data.index[i]] + row)
                        tabular.add_hline()


    def add_correlation_matrix(self):
        '''
        This method adds a table with the dataset's correlation matrix

        Not sure if we need this
        '''
        pass
    
    def add_scatter_plots(self):
        '''
        This method adds scatter plots for columns in the dataset

        Not sure if we need this
        '''
        pass

    def add_bar_charts(self):
        '''
        This method adds bar charts for each categorical column in the dataset
        '''
        import os
        import seaborn as sns
        import matplotlib.pyplot as plt

        categorical_columns = []
        for column in self.dataset.columns:
            if (self.dataset[column].dtype == 'object' and self.dataset[column].nunique() < 10) or (self.dataset[column].dtype == 'category' and self.dataset[column].nunique() < 10) or (self.dataset[column].dtype == 'bool'):
                categorical_columns.append(column)

        if len(categorical_columns) == 0:
            return
        
        # Ensure the directory exists
        os.makedirs('EDA/bar_charts', exist_ok=True)

        with self.doc.create(Subsection('Bar Charts for Categorical Columns and Histograms for Numerical Columns')):
            for column in categorical_columns:
                plt.figure(figsize=(12, 6))
                # Set larger font sizes for titles and labels
                plt.title(f'Bar Chart for {column}', fontsize=20)
                plt.xlabel(column, fontsize=18)
                plt.ylabel('Count', fontsize=18)

                # Plot the bar chart with Seaborn
                sns.countplot(data=self.dataset, x=column, color="#581845", edgecolor='black', order=self.dataset[column].value_counts().index)

                # Set tick label font size
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)

                # Save the bar chart
                image_path = f'EDA/bar_charts/{column}_bar_chart.png'
                plt.savefig(image_path)
                plt.close()

                # Include the image in the LaTeX document
                with self.doc.create(Figure(position='h!')) as fig:
                    fig.add_image(image_path, width='200px')
                    fig.add_caption(f'Bar Chart of {column}')


    def add_histograms(self):
        '''
        This method adds histograms for each numerical column in the dataset
        '''
        numerical_columns = []
        for column in self.dataset.columns:
            if str(self.dataset[column].dtype).find('int')!=-1 or str(self.dataset[column].dtype).find('float')!=-1:
                numerical_columns.append(column)

        if len(numerical_columns) == 0:
            return
        
        # Ensure the directory exists
        os.makedirs('EDA/histograms', exist_ok=True)

        # find numerical columns
        for column in numerical_columns:
            # Create a histogram for the column
            plt.figure(figsize=(12, 6))
            self.dataset[column].plot(kind='hist', bins=20, color='#C70039', edgecolor='black')
            plt.title(f'Histogram for {column}', fontsize=16)
            plt.xlabel(column, fontsize=18)
            plt.ylabel('Frequency', fontsize=18)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            
            # Save the histogram
            image_path = f'EDA/histograms/{column}_histogram.png'
            plt.savefig(image_path)
            plt.close()

            # Include the image in the LaTeX document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image(image_path, width='200px')
                fig.add_caption(f'Histogram of {column}')
                        

    def generate_report(self):
        '''
        This method generates the report
        '''
        self.add_title()  # Add the title to the report

        self.add_info_table()  # Add dataset info() table

        self.doc.append(NoEscape(r'\newpage'))
        self.add_describe_info()  # Add dataset describe() table

        self.doc.append(NoEscape(r'\newpage'))
        self.add_bar_charts()  # Add bar charts for categorical columns

        # self.add_correlation_matrix()
        # self.add_scatter_plots()

        self.add_histograms()
        
        self.doc.generate_pdf('report', clean_tex=False)

