import torch as t
import pdb

# Clear the console
print("\033c")

# Load the test file using CPU
tokens = t.load("test/test.z", map_location=t.device('cpu'))[2][0,:]

# Print the first 100 tokens
print("Tokens (top 100):")
print(tokens[:100])

# Create a tensor of unique tokens and their counts
unique_tokens = t.unique(tokens, return_counts=True)

# Sort the tokens by their counts, in descending order
sorted_tokens = unique_tokens[0][t.argsort(unique_tokens[1], descending=True)]

# Get the most common 2048 English words
import wordfreq
words = wordfreq.top_n_list('en', 2048)

# Turn the tokens into words, using the most common English word for the most common token, etc.
word_list = []
for token in tokens:
  index = sorted_tokens.tolist().index(token)
  # if index is 0, replace with '\n'. if 1, replace with '.'. if 2, replace with ','.
  # Otherwise, replace with the most common word with index - 3.
  if index == 0:
    word_list.append("\n")
  elif index == 1:
    word_list.append(".")
  elif index == 2:
    word_list.append(",")
  else:
    word_list.append(" " + words[index - 3])

text = "".join(word_list)

print("Text:")
print(text)