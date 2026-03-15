"""
In a shell, download the data to the base folder:

>> wget -nc --no-check-certificate https://course.ccs.neu.edu/cs6120s26/data/arxiv/arxiv_titles.txt
"""

import re
import string
from tqdm import tqdm
import numpy as np
import pickle
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
import seaborn as sns
import matplotlib.pyplot as plt


# @title Utility Functions

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def save_models(word_freqs, Vsvd, Vw2v,
                word2index=None, index2word=None, losses_w2v=None,
                output_dir='.'):
    '''
    Save all the appropriate data. We are expecting the variables
    shown below.
    '''
    data = {
        'word_freqs': word_freqs,
        'Vsvd': Vsvd,
        'Vw2v': Vw2v,
        'word2index': word2index,
        'index2word': index2word,
        'losses_w2v': losses_w2v,
    }

    import os
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'assignment5.pkl'), 'wb') as f:
        pickle.dump(data, f)


# @title Print Top K Word Vectors from Vectors Matrix
def print_topk(word, V, word2index, index2word, k=10):
    '''
    Args:
      - word: a string, e.g., "neural" to look up
      - V: a matrix of vectors, size num_words x embedding_size
      - word2index: dictionary of {word: index}
      - index2word: dictionary of {index: word}
      - k: number of words to print
    '''
    print("Top ", k, " words closest to ", word, ":")

    # Normalize the vectors
    V_normalized = V / np.linalg.norm(V, axis=1, keepdims=True)

    correlations = V_normalized[word2index[word]] @ V_normalized.T
    topk_indices = np.argsort(correlations)[-k:][::-1]
    for index in topk_indices:
        print(index2word[index], correlations[index])


# @title Question 1

def process_data(filename, min_cnt, max_cnt, min_words=5, min_letters=3):
    '''
    Preprocesses and builds the distribution of words in sorted order
    (from maximum occurrence to minimum occurrence) after reading the
    file. Preprocessing will include filtering out:
    * words that have non-letters in them,
    * words that are too short (under minletters)

    Arguments:
        - filename: name of file
        - min_cnt: min occurrence of words to include
        - max_cnt: max occurrence of words to include
        - min_win: minimum number of words in a title after word filtering
        - min_letters: min length of words to include (3)

    Returns:
        - word_freqs: A sorted (max to min) list of tuples of form -

            [(word1, count1), (wordN, countN), ... (wordN, countN)]

        - dataset: A list of strings with OOV words removed -

            ["this is title 1", "this is title 2", ...]
    '''
    # Read all titles from file
    with open(filename, 'r') as f:
        titles = f.readlines()

    # First pass: count word frequencies across all titles
    word_counts = {}
    all_titles_words = []  # store processed words per title

    for title in titles:
        # Lowercase the title and split into words
        words = title.strip().lower().split()
        filtered_words = []
        for word in words:
            # Strip punctuation from ends, then filter non-alphabetical words
            word_clean = word.strip(string.punctuation)
            if word_clean.isalpha() and len(word_clean) >= min_letters:
                filtered_words.append(word_clean)
        all_titles_words.append(filtered_words)

        # Count each word
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Build word_freqs: filter by count range, sorted descending by count.
    # Python sorted() is stable, and dict preserves insertion order (Python 3.7+),
    # so ties are broken by first-appearance order in the file.
    word_freqs = [(word, count) for word, count in word_counts.items()
                  if count >= min_cnt and count <= max_cnt]
    word_freqs = sorted(word_freqs, key=lambda x: -x[1])

    vocab = set(word for word, count in word_freqs)

    # Second pass: build filtered dataset
    dataset = []
    for filtered_words in all_titles_words:
        # Keep only vocabulary words
        title_vocab_words = [w for w in filtered_words if w in vocab]
        # Only keep titles with enough words
        if len(title_vocab_words) >= min_words:
            dataset.append(' '.join(title_vocab_words))

    return word_freqs, dataset


def process_data_test():
    data_test = [
        "A Case for Neural Networks Models: Deeper is Better",
        "Large Language models: Transformers require Data",
        "A United Framework of large language models",
        "A Survey of Neural Translation Models for Large Language Models"
    ]

    with open("test.txt", 'w') as f:
        for title in data_test:
            f.write(title + '\n')

    word_freqs, filtered_dataset = process_data(
        "test.txt", 2, 100, min_words=4, min_letters=4)

    # Expected data after filtering should have the properties that:
    #   - Words appear at least twice, (i.e., models, large, language, neural)
    #   - There are at least four words in the title
    #   - Each word is at least four letters long
    expected_word_freqs = [
        ('models', 5), ('large', 3), ('language', 3), ('neural', 2)
    ]

    expected_filtered_dataset = ['neural models large language models']

    assert word_freqs == expected_word_freqs, "Your word frequencies are incorrect. " \
                                              "Expected:\n" + str(expected_word_freqs) + "\nReceived:\n" + str(
        word_freqs)
    assert filtered_dataset == expected_filtered_dataset, "Your filtered dataset is incorrect. " \
                                                          "Expected:\n" + str(
        expected_filtered_dataset) + "\nReceived:\n" + str(filtered_dataset)

    print("\nprocess_data: \033[1;32mtests OK.\033[0m")


# @title Question 2.1

def create_adjacency(dataset, word2index, win=10):
    '''
    Builds an adjacency matrix based on word co-occurrence within a window.

    Args:
        - dataset: List of processed titles
        - word2index: Dictionary mapping word to index
        - win: The window size for co-occurrence.

    Returns:
        - adjacency_matrix: A NumPy array representing the adjacency matrix.
    '''
    n = len(word2index)
    adjacency_matrix = np.zeros((n, n), dtype=np.float64)
    half_win = win // 2

    for title in dataset:
        words = title.split()
        for i, w1 in enumerate(words):
            if w1 not in word2index:
                continue
            idx1 = word2index[w1]
            # Look at words within half_win distance
            start = max(0, i - half_win)
            end = min(len(words), i + half_win + 1)
            for j in range(start, end):
                if j == i:
                    continue
                w2 = words[j]
                if w2 not in word2index:
                    continue
                idx2 = word2index[w2]
                adjacency_matrix[idx1][idx2] += 1

    return adjacency_matrix


def create_adjacency_test():
    # Test file
    data_test = [
        "A Case for Neural Networks Models: Deeper is Better",
        "Large Language models: Transformers require Data",
        "A United Framework of Language Models"
    ]
    with open("test.txt", 'w') as f:
        for title in data_test:
            f.write(title + '\n')

    # Obtain word frequencies and dataset
    word_freqs, filtered_dataset = process_data(
        "test.txt", 1, 100, min_words=3, min_letters=3)

    # Create word2index to use in calculating adjacency matrix
    word2index = {word[0]: i for i, word in enumerate(word_freqs)}

    # Run your code
    adjacency_matrix = create_adjacency(filtered_dataset, word2index, win=5)

    # Since we have the option to have counts or boolean, just check boolean
    adjacency_matrix = np.array((adjacency_matrix > 0).astype(int))

    # Expected adjacency matrix
    adjacency_matrix_expected = np.array([
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    # There should be no difference
    difference = sum(sum(abs(adjacency_matrix - adjacency_matrix_expected)))
    assert difference == 0, "Your adjacency matrix is incorrect. \nExpected:\n" + \
                            str(adjacency_matrix_expected) + "\nReceived:\n" + str(adjacency_matrix) + \
                            "\n\nDifference:" + str(difference)

    # Tests OK
    print("create_adjacency: \033[1;32mtests OK.\033[0m")


# @title Question 2.2

import pandas as pd


def train_svd(adjacency_matrix, min_sv_index=3, max_sv_index=103):
    """
    Creates an embedding space using SVD on the adjacency matrix. The two parameters
    min_sv_index and max_sv_index provide the embedding size, where

        embedding_size = max_sv_index - min_sv_index.

    So, if s is a vector of all the singular values sorted from largest to smallest,
    then the embedding matrix will use the vectors corresponding to

        singular_values = s[min_sv_index:max_sv_index]

    Args:
        - adjacency_matrix: The adjacency matrix.
        - min_sv_index: The index of the largest singular value to use.
        - max_sv_index: The index of the largest singular value to use

    Returns:
        - A NumPy array representing the embedding space (num_words x embedding_dim)
    """
    # Number of singular values to compute
    # We need singular values at indices min_sv_index through max_sv_index (inclusive)
    k = min(max_sv_index + 1, min(adjacency_matrix.shape) - 1)

    # Convert to sparse matrix for svds
    A_sparse = sparse.csr_matrix(adjacency_matrix.astype(np.float64))
    U, s, Vt = svds(A_sparse, k=k)

    # svds returns singular values in ascending order, so reverse to get descending
    idx = np.argsort(-s)
    U = U[:, idx]
    s = s[idx]

    # Select the components from min_sv_index to max_sv_index (inclusive)
    U_selected = U[:, min_sv_index:max_sv_index + 1]
    s_selected = s[min_sv_index:max_sv_index + 1]

    # Weight U by singular values to get embedding
    embedding = U_selected * s_selected[np.newaxis, :]

    return embedding


def train_svd_test():
    vocabulary = ['models', 'language', 'case', 'for', 'neural', 'networks',
                  'deeper', 'better', 'large', 'transformers', 'require', 'data',
                  'united', 'framework']

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]).astype(np.float32)

    # Run your code
    U_fast = train_svd(adjacency_matrix, min_sv_index=1, max_sv_index=8)

    word_vectors = dict()
    for i, word in enumerate(vocabulary):
        word_vectors[word] = U_fast[i]

    # Check the sizes of the matrices
    assert adjacency_matrix.shape == (14, 14)
    assert U_fast.shape == (14, 8), f"Expected (14, 8), got {U_fast.shape}"

    print("train_svd: \033[1;32mtests OK.\033[0m")


# @title Question 3.1

def sample_w2v(data, word2index, neg_samples=5, win=10, sampling_distro=None):
    '''
    Randomly samples a title and a window within that title, returning
    one-hot and multi-hot vectors.

    Args:
        - data: A list of preprocessed titles.
        - word2index: A dictionary of words and their indices
        - neg_samples: Number of negative samples
        - win: The size of the context window.
        - sampling_distro: Sampling distribution of the words, likely Zipfian

    Returns:
        - wi: target vector index
        - wo: context vector index
        - Wn: negative vectors index
    '''
    vocab_size = len(word2index)

    # Randomly sample a title
    title = random.choice(data)
    words = title.split()

    # Randomly sample a target word position
    target_pos = random.randint(0, len(words) - 1)
    target_word = words[target_pos]
    wi = word2index[target_word]

    # Sample a positive context word from within the window
    half_win = win // 2
    start = max(0, target_pos - half_win)
    end = min(len(words), target_pos + half_win + 1)

    # Get context positions (exclude target itself)
    context_positions = [p for p in range(start, end) if p != target_pos]

    # Pick one random context word
    context_pos = random.choice(context_positions)
    context_word = words[context_pos]
    wo = word2index[context_word]

    # Sample negative words
    if sampling_distro is not None:
        Wn = np.random.choice(vocab_size, size=neg_samples, p=sampling_distro)
    else:
        Wn = np.random.randint(0, vocab_size, size=neg_samples)

    return wi, wo, Wn


# @title Question 3.3

def w2vgrads(vi, vo, Vns):
    """
    This function implements the gradient for all vectors in
    input matrix Vi and output matrix Vo.

    Computes gradients of LOSS L = -J where:
      J = log(sigmoid(vi^T vo)) + sum_k log(sigmoid(-vi^T vn_k))

    Args:
      - vi:  Vector of shape (d,), a sample in the input word
           vector matrix
      - vo:  Vector of shape (d,), a positive sample in the output
           word vector matrix
      - Vns: Vector of shape (k, d), k negative samples in the
           output word vector matrix

    Returns:
      - dvi, dvo, dVns: the gradients of L=-J with respect to vi, vo, vns
    """
    # Gradient of loss L = -J
    # dL/dvi = -(1 - sigmoid(vi^T vo)) * vo + sum_k sigmoid(vi^T vn_k) * vn_k
    # dL/dvo = -(1 - sigmoid(vi^T vo)) * vi
    # dL/dvn_k = sigmoid(vi^T vn_k) * vi

    sig_pos = sigmoid(np.dot(vi, vo))  # sigmoid(vi^T vo)

    # dL/dvi
    dvi = -(1 - sig_pos) * vo
    for k in range(Vns.shape[0]):
        sig_neg = sigmoid(np.dot(vi, Vns[k]))  # sigmoid(vi^T vn_k)
        dvi += sig_neg * Vns[k]

    # dL/dvo
    dvo = -(1 - sig_pos) * vi

    # dL/dVns
    dVns = np.zeros_like(Vns)
    for k in range(Vns.shape[0]):
        sig_neg = sigmoid(np.dot(vi, Vns[k]))
        dVns[k] = sig_neg * vi

    return dvi, dvo, dVns


def w2vgrads_test():
    vi = np.array([1., 0., 0., 0.])
    vo = np.array([0., 1., 0., 0.])

    Vns = np.array([
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]
    ])

    dvi, dvo, dVns = w2vgrads(vi, vo, Vns)

    expected_dvi = np.array([0.73105858, -0.5, 0.5, 0.5])
    expected_dvo = np.array([-0.5, -0., -0., -0.])
    expected_dVns = np.array([[0.5, 0., 0., 0.],
                              [0.5, 0., 0., 0.],
                              [0.73105858, 0., 0., 0.]])

    # Added assertions to check the results
    assert np.allclose(dvi, expected_dvi), "dvi calculation is incorrect"
    assert np.allclose(dvo, expected_dvo), "dvo calculation is incorrect"
    assert np.allclose(dVns, expected_dVns), "dVns calculation is incorrect"

    print("w2vgrads: \033[1;32mtests OK.\033[0m")


# @title Question 3.4

def train_w2v(dataset, word2index, word_freqs, iters=1e6, negsamps=5,
              win=5, embedding_dim=100, learning_rate=0.01,
              vectors=None):
    """
    Creates an embedding space using Word2Vec with negative sampling.

    Args:
        - dataset: A list of preprocessed titles.
        - word2index: Dictinoary assigning word to index
        - word_freqs: A list of tuples of (word, count)
        - iters: Number of iterations to run for (default 1e6)
        - negsamps: Number of negative samples
        - win: The size of the context window for sampling.
        - embedding_dim: The desired dimensionality of the embedding space.
        - learning_rate: Learning rate or any other DNN params with defaults.
                       The autograder won't touch this.

    Returns:
        - Vw2v: a tuple with two numpy arrays of size N words x d dimensions
        - List of losses (to print out)
    """
    iters = int(iters)
    vocab_size = len(word2index)

    # Initialize embedding matrices
    if vectors is not None:
        Vi, Vo = vectors
    else:
        Vi = np.random.randn(vocab_size, embedding_dim) * 0.01
        Vo = np.random.randn(vocab_size, embedding_dim) * 0.01

    # Build sampling distribution: U(w)^(3/4) normalized
    counts = np.array([count for (word, count) in word_freqs], dtype=np.float64)
    sampling_distro = np.power(counts, 0.75)
    sampling_distro /= sampling_distro.sum()

    losses = []

    for it in tqdm(range(iters), desc="Training Word2Vec"):
        # Sample target, positive context, and negative samples
        wi, wo, Wn = sample_w2v(dataset, word2index, neg_samples=negsamps,
                                win=win, sampling_distro=sampling_distro)

        # Get vectors
        vi = Vi[wi]
        vo = Vo[wo]
        Vns = Vo[Wn]

        # Compute gradients (of loss L = -J)
        dvi, dvo, dVns = w2vgrads(vi, vo, Vns)

        # Gradient descent on L (= gradient ascent on J)
        Vi[wi] -= learning_rate * dvi
        Vo[wo] -= learning_rate * dvo
        Vo[Wn] -= learning_rate * dVns

        # Compute loss (positive part for tracking)
        dot_pos = np.dot(Vi[wi], Vo[wo])
        loss = -np.log(sigmoid(dot_pos) + 1e-10)
        losses.append(loss)

    return (Vi, Vo), losses


def train_w2v_test():
    vocabulary = ['models', 'language', 'case', 'for', 'neural', 'networks',
                  'deeper', 'better', 'large', 'transformers', 'require', 'data',
                  'united', 'framework']

    word2index = {word: i for i, word in enumerate(vocabulary)}

    data_test = [
        "A Case for Neural Networks Models: Deeper is Better",
        "Large Language models: Transformers require Data",
        "A United Framework of Language Models"
    ]
    with open("test.txt", 'w') as f:
        for title in data_test:
            f.write(title + '\n')

    word_freqs, filtered_dataset = process_data(
        "test.txt", 1, 100, min_words=3, min_letters=3)

    (Vw2v_Vi, Vw2v_Vo), losses = train_w2v(filtered_dataset, word2index, word_freqs,
                                           iters=1000, negsamps=5, win=3,
                                           embedding_dim=100, learning_rate=0.01)

    assert Vw2v_Vi.shape == (14, 100)
    assert Vw2v_Vo.shape == (14, 100)
    assert len(losses) == 1000


# @title Main Function

if __name__ == '__main__':

    # Run unit tests
    process_data_test()
    create_adjacency_test()
    train_svd_test()
    w2vgrads_test()
    train_w2v_test()

    # ===== Path Configuration =====
    import os

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "as5_file")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "as5_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process "arxiv_titles.txt". Download from:
    #    https://course.ccs.neu.edu/cs6120s26/data/arxiv/arxiv_titles.txt
    word_freqs, filtered_dataset = process_data(
        os.path.join(DATA_DIR, "arxiv_titles.txt"), 100, 1e10, min_words=5, min_letters=4)

    # Create the appropriate data structures from word_freqs variable.
    vocabulary = [tuple[0] for tuple in word_freqs]
    word2index = {word: i for i, word in enumerate(vocabulary)}
    index2word = {i: word for i, word in enumerate(vocabulary)}

    # ======== Q1: Plot Zipfian Distribution ========
    counts = [c for (w, c) in word_freqs]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(counts)), counts)
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    plt.title("Zipfian Distribution of Word Frequencies")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "zipfian_distribution.png"), dpi=150)
    plt.close()
    print(f"Q1: Vocabulary size = {len(word_freqs)}, Dataset size = {len(filtered_dataset)}")
    print("Q1: Zipfian distribution saved to zipfian_distribution.png")

    # ======== Q2.1: Create Adjacency Matrix ========
    adjacency_matrix = create_adjacency(filtered_dataset, word2index, win=5)
    print(f"Q2.1: Adjacency matrix shape = {adjacency_matrix.shape}")

    # ======== Q2.2: Train SVD Embeddings ========
    Vsvd = train_svd(adjacency_matrix, min_sv_index=1, max_sv_index=100)
    print(f"Q2.2: SVD embedding shape = {Vsvd.shape}")

    # Print nearest neighbors for SVD
    test_words = ["neural", "machine", "dark", "string", "black"]
    print("\n" + "=" * 60)
    print("Q2.2: SVD Nearest Neighbors")
    print("=" * 60)
    for word in test_words:
        if word in word2index:
            print_topk(word, Vsvd, word2index, index2word, k=10)
            print()
        else:
            print(f"'{word}' not in vocabulary\n")

    # ======== Q3.4: Train Word2Vec ========
    Vw2v, losses = train_w2v(filtered_dataset, word2index, word_freqs, iters=1e6, negsamps=1,
                             win=5, embedding_dim=100, learning_rate=0.01, vectors=None)

    # Plot loss function
    plt.figure(figsize=(10, 6))
    # Smooth losses by averaging every 1000 iterations
    window_size = 1000
    smoothed_losses = [np.mean(losses[i:i + window_size])
                       for i in range(0, len(losses) - window_size + 1, window_size)]
    plt.plot(range(len(smoothed_losses)), smoothed_losses)
    plt.xlabel("Iteration (x1000)")
    plt.ylabel("Loss (positive part)")
    plt.title("Word2Vec Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "w2v_loss.png"), dpi=150)
    plt.close()
    print("Q3.4: Loss curve saved to w2v_loss.png")

    # Print nearest neighbors for Word2Vec
    print("\n" + "=" * 60)
    print("Q3.4: Word2Vec Nearest Neighbors")
    print("=" * 60)
    for word in test_words:
        if word in word2index:
            print_topk(word, Vw2v[0], word2index, index2word, k=10)
            print()
        else:
            print(f"'{word}' not in vocabulary\n")

    # ======== Save all models ========
    save_models(word_freqs, Vsvd, Vw2v, word2index, index2word, losses, output_dir=OUTPUT_DIR)
    print(f"All models saved to {OUTPUT_DIR}/assignment5.pkl")