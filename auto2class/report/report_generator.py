from pylatex import Command, Document, Section, Subsection, Tabular
from pylatex.utils import NoEscape
from pylatex.table import Table 
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pylatex import Section, Subsection, Figure, NoEscape
import seaborn as sns
import matplotlib.pyplot as plt
from .plot_generator import PlotGenerator

class ReportGenerator:
    def __init__(self, dataset, dataset_name, op):
        self.doc = Document()
        self.dataset = pd.DataFrame(dataset)
        self.dataset_name = dataset_name


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

        with self.doc.create(Section('Exploratory Data Analysis Part')):
            with self.doc.create(Subsection('DataTypes and Non-Null Count')):
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
            self.new_page()
            # Capture dataset.describe() into a buffer
            describe_data = self.dataset.describe().transpose()

            # Extract header and rows for descriptive statistics
            header = list(describe_data.columns)
            table_rows = describe_data.values.tolist()

            with self.doc.create(Subsection('Descriptive Statistics')):
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

    def add_bar_charts(self):
        '''
        This method adds bar charts for each categorical column in the dataset
        '''
        self.new_page()

        plt = PlotGenerator().generate_bar_charts(self.dataset)
        if plt is None:
            return
        
        with self.doc.create(Subsection('Bar Charts of Categorical columns')):
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image('EDA/bar_charts.png', width='460px')
                fig.add_caption('Bar Charts of Categorical columns')


    def add_histograms(self):
        '''
        This method adds histograms for each numerical column in the dataset
        '''
        self.new_page()
        plt = PlotGenerator().generate_histograms(self.dataset)
        if plt is None:
            return
        
        with self.doc.create(Subsection('Histograms of Numerical columns')):
            # put image in the latex document
            with self.doc.create(Figure(position='h!')) as fig:
                fig.add_image('EDA/histograms.png', width='460px')
                fig.add_caption('Histograms of Numerical columns')
        return 
                        

    def generate_report(self):
        '''
        This method generates the report
        '''
        self.make_small_margins()  # Reduce the margins of the document
        self.add_title()  # Add the title to the report
        self.add_table_of_contents()  # Add table of contents

        self.add_info_table()  # Add dataset info() table
        self.add_describe_info()  # Add dataset describe() table

        self.add_histograms() # Add histograms for numerical columns
        self.add_bar_charts() # Add bar charts for categorical columns
        
        self.doc.generate_pdf('report', clean_tex=False)

