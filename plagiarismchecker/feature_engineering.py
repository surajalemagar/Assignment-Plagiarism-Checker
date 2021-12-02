# import libraries
from typing import DefaultDict
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# csv_file = 'plagiarismchecker/data/file_information.csv'
csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print(plagiarism_df.head())

# Define function to Convert all Category labels to numerical labels according to the following rules
# (a higher value indicates a higher degree of plagiarism):
# 0 = non
# 1 = heavy
# 2 = light
# 3 = cut
# -1 = orig, this is a special value that indicates an original file.

def num_cat(x):
    if x == 'non':
        return 0
    elif x == 'heavy':
        return 1
    elif x == 'light':
        return 2
    elif x == 'cut':
        return 3
    elif x == 'orig':
        return -1

# Define a function to create a new 'Class' column as per following statements:
# Any answer text that is not plagiarized (non) should have the class label 0.
# Any plagiarized answer texts should have the class label 1.
# And any orig texts will have a special label -1.

def col_class(x):
    if x == 'non':
        return 0
    elif x in ['heavy','light','cut']:
        return 1
    elif x == 'orig':
        return -1

# Read in a csv file and return a transformed dataframe
def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things: 
       1) converts `Category` column values to numerical values 
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''
    
    df = pd.read_csv(csv_file)
    
    # Use function num_cat & col_class
    df['Category_new'] = df['Category'].apply(lambda x: num_cat(x))
    df['Class'] = df['Category'].apply(lambda x: col_class(x))
    
    # Drop original column 'Category' and rename column 'Category_new' to 'Category'
    df.drop(['Category'],axis=1, inplace=True)
    df.rename(columns={'Category_new': 'Category'}, inplace=True)
    
    return df

# informal testing, print out the results of a called function
# create new `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
# check that all categories of plagiarism have a class label = 1
# print(transformed_df.head(10))

# test cell that creates `transformed_df`, if tests are passed

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

# importing tests
from plagiarismchecker import problem_unittests as tests
#import problem_unittests as tests

# test numerical_dataframe function
tests.test_numerical_df(numerical_dataframe)

# if above test is passed, create NEW `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
# print('\nExample data: ')
# print(transformed_df.head())

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# from plagiarismchecker import helpers 
from . import helpers 
# create a text column 
text_df = helpers.create_text_column(transformed_df)
# print(text_df.head())

# after running the cell above
# check out the processed text for a single file, by row index
row_idx = 0 # feel free to change this index

sample_text = text_df.iloc[0]['Text']

# print('Sample processed text:\n\n', sample_text)

random_seed = 1 # can change; set for reproducibility

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# from plagiarismchecker import helpers
from . import helpers 


# create new df with Datatype (train, test, orig) column
# pass in `text_df` from above to create a complete dataframe, with all the information you need
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# check results
# print(complete_df.head(10))

def containment(ngram_array):
    ''' Containment is a measure of text similarity. It is the normalized, 
       intersection of ngram word counts in two texts.
       :param ngram_array: an array of ngram counts for an answer and source text.
       :return: a normalized containment value.'''
    # the intersection can be found by looking at the columns in the ngram array
    # this creates a list that holds the min value found in a column
    # so it will hold 0 if there are no matches, and 1+ for matching word(s)
    intersection_list = np.amin(ngram_array, axis=0)
    
    # optional debug: uncomment line below
    # print(intersection_list)

    # sum up number of the intersection counts
    intersection = np.sum(intersection_list)
    
    # count up the number of n-grams in the answer text
    answer_idx = 0
    answer_cnt = np.sum(ngram_array[answer_idx])
    
    # normalize and get final containment value
    containment_val =  intersection / answer_cnt
    
    return containment_val

# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text, 
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    
    # your code here
    source_filename = 'orig_' + answer_filename.split('_')[1]
    a_text = df[df['File'] == answer_filename]['Text'].values[0]
    s_text = df[df['File'] == source_filename]['Text'].values[0]


    # instantiate an ngram counter
    counts = CountVectorizer(analyzer='word', ngram_range=(n,n))
    
    # create array of n-gram counts for the answer and source text
    ngrams = counts.fit_transform([a_text, s_text])
    ngram_array = ngrams.toarray()
    
    
    return containment(ngram_array)

print(calculate_containment(complete_df, 1, 'g0pB_taskd.txt'))
# # select a value for n
# n = 3

# # indices for first few files
# test_indices = range(5)

# # iterate through files and calculate containment
# category_vals = []
# containment_vals = []
# for i in test_indices:
#     # get level of plagiarism for a given file index
#     category_vals.append(complete_df.loc[i, 'Category'])
#     # calculate containment for given file and n
#     filename = complete_df.loc[i, 'File']
#     c = calculate_containment(complete_df, n, filename)
# #     containment_vals.append(c)

# # # print out result, does it make sense?
# # print('\nOriginal category values: \n', category_vals)
# # print()
# # print(str(n)+'-gram containment values: \n', containment_vals)


# # # Saving the model
# # filename ='finalized_model.sav'
# # joblib.dump(calculate_containment(complete_df, 1, 'g0pB_taskd.txt'), filename)
