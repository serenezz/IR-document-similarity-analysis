'''
CSC790 Assignment 3
Serene Zan

Instructions:
Open the terminal in the 'HW03 Zan' folder.
I added a few more special characters to remove and renamed the file to chars.txt.
Please type 'python 3.py 10' to run the assignement.
Can change the number k in the CLI for any desired number.
'''


import os
import sys
import nltk as tk
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from collections import Counter
import math



def read_file(file_path):
    '''
    Read strings from the file and store in a list.
    Parameters:
    1. file_path : str
        Path of the file.
    Returns:
    1. doc : list
        A list of strings read from the file.
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        doc = file.read().splitlines()

    return doc



def read_docs_from_folder(folder_path):
    '''
    Read documents from the folder and store in a list.
    Parameters:
    1. folder_path : str
        Path of the folder containing the docs.
    Returns:
    1. docs : dicitonary
        A dictionary: key - file number, value - doc strings read from the file.
    '''
    docs = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
                # Extract the number from file_1.txt
                number = file_name.split('.')[0].split('_')[-1]
                docs[number] = file.read()

    return docs



def tokenization(doc):
    '''
    Tokenize the string by calling the nltk method.
    Parameters:
    1. doc : string
        A text string.
    Returns:
    1. tokens : list
        A list of tokenzied terms.
    '''
    tokens = tk.word_tokenize(doc)
    return tokens



def removing(tokens, to_be_removed):
    '''
    Remove stop words/punctuation from the list of tokens.
    Parameters:
    1. tokens : list
        A list of tokenized terms.
    2. to_be_removed : list
        A list of stopwords/punctuation to be removed.
    Returns:
    1. processed_tokens : list
        A list of tokenized terms without stop words.
    '''
    processed_tokens = [word for word in tokens if not word in to_be_removed]
    return processed_tokens



def stemming(tokens):
    '''
    Stemming the tokens to match.
    Parameters:
    1. tokens : list
        A list of terms.
    Returns:
    1. stem_tokens : list
        A list of tokenized and stemmed terms without stopwords or punctuations.
    '''
    stem_tokens = [tk.stem.PorterStemmer().stem(word) for word in tokens]
    return stem_tokens



def preprocess_doc(docs, stop_words, punctuation):
    '''
    Preprocess the docs by calling functions for tokenization, stemming, and removing stopwords and punctuations.
    Parameters:
    1. docs : dictionary
        A dictionary of strings, each string is a document.
    2. stop_words : list
        A list of stop words.
    3. punctuation : list
        A list of punctuations.
    Returns:
    1. docs_processed : list
        A list of lists, each list contains the preprocessed terms of each document.
    '''
    docs_processed = []
    for doc in docs.values():
        # Tokenize
        doc = tokenization(doc)
        # Convert to lowercase
        doc = [word.lower() for word in doc]
        # Remove punctuation
        doc = removing(doc, punctuation)
        # Remove stop words
        doc = removing(doc, stop_words)
        # Stemming
        doc_processed = stemming(doc)
        docs_processed.append(doc_processed)

    return docs_processed



def calculate_tf(docs):
    '''
    Create a Dataframe for the tf, with doc number as row and term as column.
    Parameters:
    1. docs : list
        A list of lists, each internal list stores all the preprocessed tokens from each doc.
    Returns:
    1. docs_tf : Dataframe
        A Dataframe representing tf.
    '''
    docs_tf = pd.DataFrame()
    # Count tf
    counts = [Counter(doc) for doc in docs]
    # Dataframe for tf
    docs_tf = pd.DataFrame(counts)
    # Modify the values and type
    docs_tf = docs_tf.fillna(0).astype(int)
    # Moodify the index range
    docs_tf.index = range(1, len(docs_tf)+1)
    
    # print(docs_tf)
    return docs_tf



def find_intersection(docs_tf):
    '''
    Find the intersection terms between two documents.
    Parameters:
    1. docs_tf : Dataframe
        A Dataframe representing tf.
    Returns:
    1. intersection : dictionary
        A dictionary of intersection words, with tuples of document number pairs as key.
    '''
    intersection = {}
    for i in range(1, len(docs_tf)+1):
        for j in range(i + 1, len(docs_tf)+1):
            # print(f'i {i}, j {j}')
            # Retrive words present in each document
            doc1 = set(docs_tf.columns[docs_tf.loc[i] != 0])
            doc2 = set(docs_tf.columns[docs_tf.loc[j] != 0])
            # Then obtain the intersection
            common_terms = doc1.intersection(doc2)
            terms = []
            for term in common_terms:
                terms.append(term)
            # Append using document number tuple as key
            intersection[(i, j)] = terms

    # print(intersection)
    return intersection
            


def comp1_tf(docs_tf, intersection):
    '''
    Compute the cos similarity of every document pair using tf.   
    Parameters:
    1. docs_tf : Dataframe
        A Dataframe representing tf.
    2. intersection : dictioinary
        A dictionary of intersection words.
    Returns:
    1. result_tf : dictionary
        A dictionary of resulting cos simiarity, with tuples of document pairs as key.
    '''
    tf_pairs = {}
    # Go through the common words for vector
    for pair, terms in intersection.items():
        doc1, doc2 = pair
        # print(f'i: {doc1}, j: {doc2}, List value: {terms}')
        if terms == []:
            tf_pairs[pair] = 0
            continue
        # Obtain tf for each term
        tf_1_2 = []
        for term in terms:
            tf_1 = docs_tf.loc[doc1, term]
            tf_2 = docs_tf.loc[doc2, term]
            tf_1_2.append([tf_1, tf_2])
        tf_pairs[pair] = tf_1_2
        
    # Create dictionary storing cos similarity
    result_tf = {}
    for pair, tfs in tf_pairs.items():
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        if tfs == 0:
            result_tf[pair] = 0
            continue
        for tf in tfs:
            numerator += tf[0] * tf[1]
            denominator1 += tf[0]**2
            denominator2 += tf[1]**2
        result_tf[pair] = round(numerator / (math.sqrt(denominator1) * math.sqrt(denominator2)), 4)
    
    # result_tf = dict(sorted(result_tf.items(), key=lambda x: x[1]))
    return result_tf
    


def comp2_tf_idf(docs_tf, intersection):
    '''
    Compute the cos similarity of every document pair using tf and idf.   
    Parameters:
    1. docs_tf : Dataframe
        A Dataframe representing tf.
    2. intersection : dictioinary
        A dictionary of intersection words.
    Returns:
    1. result_tf : dictionary
        A dictionary of resulting cos simiarity, with tuples of document pairs as key.
    '''
    # Calculate df
    docs_df = (docs_tf != 0).sum(axis=0)
    docs_idf = pd.DataFrame(docs_df, columns=['df'])
    # N
    N = len(docs_tf)
    # Calculate idf
    idf = []
    for df in docs_idf['df']:
        result = math.log10(N/df)
        idf.append(result)
    docs_idf['idf'] = idf
    # Go through the common words for vector
    tf_pairs = {}
    for pair, terms in intersection.items():
        doc1, doc2 = pair
        if terms == []:
            tf_pairs[pair] = 0
            continue
        # Obtain tf for each term
        tf_1_2 = {}
        for term in terms:
            tf_1 = docs_tf.loc[doc1, term]
            tf_2 = docs_tf.loc[doc2, term]
            tf_1_2[term] = [tf_1, tf_2]
        tf_pairs[pair] = tf_1_2

    # Create dictionary storing cos similarity
    result_tf_idf = {}
    for pair, tfs in tf_pairs.items():
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        # No common words
        if tfs == 0:
            result_tf_idf[pair] = 0
            continue
        # Calculate using vector
        for term, tf in tfs.items():
            term_idf = docs_idf.loc[term, 'idf']
            numerator += tf[0] * term_idf * tf[1] * term_idf
            denominator1 += (tf[0] * term_idf) ** 2
            denominator2 += (tf[1] * term_idf) ** 2      
        result_tf_idf[pair] = round(numerator / (math.sqrt(denominator1) * math.sqrt(denominator2)), 4)    
    return result_tf_idf



def comp3_wf_idf(docs_tf, intersection):
    '''
    Compute the cos similarity of every document pair using tf and wf-idf.   
    Parameters:
    1. docs_tf : Dataframe
        A Dataframe representing tf.
    2. intersection : dictioinary
        A dictionary of intersection words.
    Returns:
    1. result_tf : dictionary
        A dictionary of resulting cos simiarity, with tuples of document pairs as key.
    '''
    # Calculate df
    docs_df = (docs_tf != 0).sum(axis=0)
    docs_idf = pd.DataFrame(docs_df, columns=['df'])
    # N
    N = len(docs_tf)
    # Calculate idf
    idf = []
    for df in docs_idf['df']:
        result = math.log10(N/df)
        idf.append(result)
    docs_idf['idf'] = idf
    tf_pairs = {}
    # Go through the common words for vector
    for pair, terms in intersection.items():
        doc1, doc2 = pair
        if terms == []:
            tf_pairs[pair] = 0
            continue
        # Obtain tf for each term
        tf_1_2 = {}
        for term in terms:
            tf_1 = docs_tf.loc[doc1, term]
            tf_2 = docs_tf.loc[doc2, term]
            tf_1_2[term] = [tf_1, tf_2]
        tf_pairs[pair] = tf_1_2

    # Create dictionary storing cos similarity
    result_wf_idf = {}
    for pair, tfs in tf_pairs.items():
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        # No common words
        if tfs == 0:
            result_wf_idf[pair] = 0
            continue
        # Calculate similarity using vector
        for term, tf in tfs.items():
            term_idf = docs_idf.loc[term, 'idf']
            numerator += (1 + math.log10(tf[0])) * term_idf * (1 + math.log10(tf[1])) * term_idf
            denominator1 += ((1 + math.log10(tf[0])) * term_idf) ** 2
            denominator2 += ((1 + math.log10(tf[1])) * term_idf) ** 2
        result_wf_idf[pair] = round(numerator / (math.sqrt(denominator1) * math.sqrt(denominator2)), 4)    
    
    return result_wf_idf



def display_info():
    '''
    Displays course and name.
    '''
    print('\n=================== CSC790-IR Homework 03 ===================')
    print('First Name: Serene')
    print('Last Name: Zan')
    print('============================================================')



def display_frequent_terms(docs_tf, k):
    '''
    Display top k frequent terms based on cf.
    Parameters:
    1. docs_tf : Dataframe
        The tf Dataframe
    2. k : integer
        The integer from CLI
    '''
    # Generate cf and sort
    docs_cf = docs_tf.sum()
    sorted_docs_cf = docs_cf.sort_values(ascending=False)

    print(f'\nThe number of unique words is: {len(docs_tf.columns)}')
    print(f'The top {k} most frequent words are:')
    for index, (term, cf) in enumerate(sorted_docs_cf.head(k).items(), start=1):
        print(f"{index}. {term}: {cf}")
    print('=======================================================')



def display_similarity(measure, k, result):
    '''
    Display the top similar document pairs based on input k
    Parameters:
    1. measure : string
        A string indicating the measure (tf, tf-idf, wf-idf).
    2. k : integer
        The number from CLI.
    3. result : dictionary
        A dictionary storing: key - tuple of doc pair, value - similarity float.
    '''
    print(f'\n{measure}')
    result_sorted_k = sorted(result.items(), key=lambda x: x[1], reverse=True)[:k]
    # Retrieve the similarity using tuple key
    for pair in result_sorted_k:
        print(f'file {pair[0][0]} and file {pair[0][1]} with similarity of {pair[1]}')



def main():
    if __name__ == '__main__':

        # Get k from CLI
        k = int(sys.argv[1])

        # Read from files
        stopwords_path = 'stopwords.txt'
        punctuation_path = 'chars.txt'
        stop_words = read_file(stopwords_path)
        punctuation = read_file(punctuation_path)
        docs = read_docs_from_folder('documents')
        docs = dict(sorted(docs.items(), key=lambda x: int(x[0])))

        # Preprocess
        docs_processed = preprocess_doc(docs, stop_words, punctuation)

        # Create tf matrix
        docs_tf = calculate_tf(docs_processed)
        # print(docs_tf)

        # Create common term for each document pair
        intersection = find_intersection(docs_tf)
        # print(intersection)

        # Multithreading        
        with ThreadPoolExecutor() as executor:  
            # Thread 1 using only tf
            thread1 = executor.submit(comp1_tf, docs_tf, intersection)
            # Thread 2 using tf-idf
            thread2 = executor.submit(comp2_tf_idf, docs_tf, intersection)
            # Thread 3 using wf-idf
            thread3 = executor.submit(comp3_wf_idf, docs_tf, intersection)

        # Multithreading results
        result_tf = thread1.result()
        result_tf_idf = thread2.result()
        result_wf_idf = thread3.result()

        # Display course info
        display_info()

        # Display top k frequent terms
        display_frequent_terms(docs_tf, k)
        
        print(f'\nThe top {k} closest documents are:')
        display_similarity('1. Using tf', k, result_tf)
        display_similarity('2. Using tf-idf', k, result_tf_idf)
        display_similarity('3. Using wf-idf', k, result_wf_idf)

        
        

main()