# Complete working script for Assignment 7
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import shifterator as sh
from collections import Counter
import numpy as np

def get_words_string(book_name):
    """Read the content of the book and return a cleaned string of words."""
    with open(book_name, 'r') as file:
        text = file.read()
    
    text = text.replace('\n', ' ').split(' ')
    text = [entry.strip().lower() for entry in text if entry.strip()]
    return ' '.join(text)

def tokenize(string):
    """Tokenizes the input text into words and punctuation, and returns a list with nested lists of individual tokens"""
    word_regex = r'\b\w+\b'
    punct_regex = r'[^\w\s]'
    dot_regex = r'\.{4,}|\.{3}|\.{2}|\.'
    
    tokens = []
    for match in re.finditer(f'{dot_regex}|{punct_regex}|{word_regex}', string):
        token = match.group()
        if re.match(word_regex, token) or re.match(dot_regex, token):
            tokens.append(token)
        else:
            parts = re.findall(f'(?:{dot_regex})+|[^\.\s]+', token)
            tokens.extend(parts)
    return tokens

def get_happiness_scores(ngrams, hedon, hap_vars):
    """Calculate happiness scores for the given ngrams using the hedonometer data."""
    hap_df = hedon[hap_vars]
    hap_dict = dict(zip(hap_df[hap_vars[0]], hap_df[hap_vars[1]]))
    return [hap_dict.get(gram, 0) for gram in ngrams]

def calculate_averages(raw_scores, window_size):
    """Calculate rolling averages of the given scores with the specified window size."""
    raw_series = pd.Series(raw_scores)
    return raw_series.rolling(window_size).mean()

def plotter(data, titles, rows, cols):
    """Plot the given data with the specified layout."""
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 10))
    fig.tight_layout()
    sns.despine(fig)
    
    for avg, ax, title in zip(data, axs.ravel(), titles):
        ax.plot(avg)
        ax.set_title(title, fontsize=7)
        ax.set_ylabel(r"$h_{avg}$", fontsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=5)
    axs.ravel()[-1].set_xlabel("Word number i")

def word_shifts(type2freq_1, type2freq_2, ref_avg, title):
    """Plot weighted average word shift graphs."""
    sentiment_shift = sh.WeightedAvgShift(
        type2freq_1=type2freq_1,
        type2freq_2=type2freq_2,
        type2score_1='labMT_English',
        reference_value=ref_avg,
        stop_lens=[(4, 6)]
    )
    sentiment_shift.get_shift_graph(detailed=True, system_names=title)

# Tokenize Star Wars
starwars_string = get_words_string('starwars_stitched.txt')
starwars_clean = tokenize(starwars_string)
print(len(starwars_clean))
print(starwars_clean.index('dismounts'))

# Tokenize LOTR... we don't use LOTR but you can if you'd like. 
# lotr_string = get_words_string('lotr_stitched.txt')
# lotr_clean = tokenize(lotr_string)
# print(len(lotr_clean))
# print(type(lotr_clean[0]))

# Read in lexicon
hap_vars = ['Word', 'Happiness Score']
hedonometer = pd.read_csv('labMT_lexicon.csv', names=hap_vars)

# Get happiness scores for Star Wars 
starwars_happiness_scores = get_happiness_scores(starwars_clean, hedonometer, hap_vars)

# Get happiness scores for LOTR
# lotr_happiness_scores = get_happiness_scores(lotr_clean, hedonometer, hap_vars)

# Text separation point for Star Wars
text_separator = 111865
prequels = starwars_clean[:text_separator]
original_trilogy = starwars_clean[text_separator:]

# Get the counts for each segment
first_freqs = dict(Counter(prequels))
second_freqs = dict(Counter(original_trilogy))

# Calculate average happiness scores
first_avg = np.average(get_happiness_scores(prequels, hedonometer, hap_vars))
second_avg = np.average(get_happiness_scores(original_trilogy, hedonometer, hap_vars))

# Plot word shifts
word_shifts(first_freqs, second_freqs, first_avg, ["Prequels", 'Original Trilogy'])
word_shifts(second_freqs, first_freqs, second_avg, ["Original Trilogy", "Prequels"])
word_shifts(second_freqs, first_freqs, 5, ["Original Trilogy", "Prequels"])
