# import libraries
import pandas as pd
import numpy as np
import os

csv_file = 'plagiarismchecker/data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
# print(plagiarism_df.head(10))
# print(plagiarism_df.shape)

# print out some stats about the data
# print('Number of files: ', plagiarism_df.shape[0])  # .shape[0] gives the rows 
# .unique() gives unique items in a specified column
# print('Number of unique tasks/question types (A-E): ', (len(plagiarism_df['Task'].unique())))
# print('Unique plagiarism categories: ', (plagiarism_df['Category'].unique()))

# Show counts by different tasks and amounts of plagiarism

# group and count by task
counts_per_task=plagiarism_df.groupby(['Task']).size().reset_index(name="Counts")
# print("\nTask:")
# print(counts_per_task)

# group by plagiarism level
counts_per_category=plagiarism_df.groupby(['Category']).size().reset_index(name="Counts")
# print("\nPlagiarism Levels:")
# print(counts_per_category)

# group by task AND plagiarism level
counts_task_and_plagiarism=plagiarism_df.groupby(['Task', 'Category']).size().reset_index(name="Counts")
# print("\nTask & Plagiarism Level Combos :")
# print(counts_task_and_plagiarism)

import matplotlib.pyplot as plt
#% matplotlib inline

# counts
group = ['Task', 'Category']
counts = plagiarism_df.groupby(group).size().reset_index(name="Counts")

# plt.figure(figsize=(8,5))
# plt.bar(range(len(counts)), counts['Counts'], color = 'blue')