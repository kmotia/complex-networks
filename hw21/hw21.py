# Complete working script for Assignment 6
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import seaborn as sns

def read_and_preprocess_file(filename):
    """Read the file and preprocess the text."""
    with open(filename, 'r') as file:
        text = file.read().replace('\n', ' ')
        
    words = [word for word in text.split(' ') if word.strip()]
    capital_words = [word.lower() for word in words if word.isupper()]
    all_words_lower = [word.lower() for word in words]
    filtered_words = [word for word in all_words_lower if word not in capital_words]
    
    return filtered_words

def tokenize(text):
    """Tokenizes the input text into words and punctuation, and returns a single list of tokens"""
    word_regex = r'\b\w+\b'
    punct_regex = r'[^\w\s]'
    dot_regex = r'\.{4,}|\.{3}|\.{2}|\.'
    
    tokens = []
    for match in re.finditer(f'{dot_regex}|{punct_regex}|{word_regex}', text):
        token = match.group()
        if re.match(word_regex, token):
            tokens.append([token])
        elif re.match(dot_regex, token):
            tokens.append([token])
        else:
            parts = re.findall(f'(?:{dot_regex})+|[^\.\s]+', token)
            for part in parts:
                tokens.append([part])
    
    return tokens

def get_happiness_scores(ngrams, hedon, hap_vars):
    """Retrieve happiness scores for the given n-grams."""
    hap_df = hedon[hap_vars]
    hap_dict = dict(zip(hap_df[hap_vars[0]], hap_df[hap_vars[1]]))
    scores = [hap_dict.get(gram, 0) for gram in ngrams]
    return scores

def calculate_averages(scores, window_size):
    """Calculate rolling averages with a specified window size."""
    series = pd.Series(scores)
    rolling_window = series.rolling(window_size)
    return rolling_window.mean()

def plot_averages(data, titles, rows, cols):
    """Plot the rolling averages."""
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 10))
    fig.tight_layout()
    sns.despine(fig)
    
    for avg, ax, title in zip(data, axs.ravel(), titles):
        ax.plot(avg)
        ax.set_title(title, fontsize=7)
        ax.set_ylabel(r"$h_{avg}$", fontsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=5)
    axs.ravel()[-1].set_xlabel("Word number i")

# Read and preprocess file
filtered_words = read_and_preprocess_file('starwars_stitched.txt')
combined_text = ' '.join(filtered_words)
tokenized_words = tokenize(combined_text)

# Convert tokenized words to list of strings
grams_list = [' '.join(inner_list) for inner_list in tokenized_words]

# Read happiness lexicon
hap_vars = ['Word', 'Happiness Score']
hedonometer = pd.read_csv('labMT_lexicon.csv', names=hap_vars)

# Get happiness scores
happiness_scores = get_happiness_scores(grams_list, hedonometer, hap_vars)

# Calculate and plot rolling averages
window_sizes = [round(10**i) for i in np.arange(1, 4.5, 0.5)]
rolling_averages = [calculate_averages(happiness_scores, ws) for ws in window_sizes]
titles = [f"Original 6 Star Wars Movies (Episode Order 1-6), T = {ws}, z = {round(np.log10(ws)*2)/2}" for ws in window_sizes]
plot_averages(rolling_averages, titles, 7, 1)
plt.savefig("unadjusted_happiness.png")
plt.show()

# Adjusted happiness scores
avg_happiness = np.average(happiness_scores)
delta_h_list = np.arange(0.5, 4, 0.5)
blocked_happiness = [(avg_happiness - delta, avg_happiness + delta) for delta in delta_h_list]
adjusted_happiness = [
    [score for score in happiness_scores if (block[0] >= score or score >= block[1])]
    for block in blocked_happiness
]

# Plot adjusted rolling averages
optimal_window_size = 1000
adjusted_rolling_averages = [calculate_averages(scores, optimal_window_size) for scores in adjusted_happiness]
delta_titles = [f"T = 1000, Δh = {delta}" for delta in delta_h_list]
plot_averages(adjusted_rolling_averages, delta_titles, 7, 1)
plt.savefig("varying_deltas.png")
plt.show()


