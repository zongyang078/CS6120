# Zongyang Li
# li.zongyang@northeastern.edu

import nltk
import math
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# Special Tokens
class SpecialTokens:
    """Container for special tokens used in the language model."""

    def __init__(self, start_token="<s>", end_token="<e>", unknown_token="<unk>"):
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token


# Question 1: Preprocessing and Vocabulary
def read_and_tokenize_sentences(filename, sample_delimiter='\n'):
    """
    Read data from file and tokenize each sentence using NLTK.

    Each line (delimited by sample_delimiter) is treated as one sentence.
    Sentences are lowercased and tokenized into word-level tokens.

    Args:
        filename: Path to the text file.
        sample_delimiter: Delimiter used to split text into sentences.

    Returns:
        tokenized_data: A list of lists, where each inner list contains
                        the lowercased tokens of one sentence.
                        e.g. [["hello", "world", "!"], ["another", "tweet"]]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into sentences by the delimiter
    sentences = text.split(sample_delimiter)

    tokenized_data = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip empty lines
        if not sentence:
            continue
        # Lowercase and tokenize with NLTK
        tokens = nltk.word_tokenize(sentence.lower())
        if tokens:
            tokenized_data.append(tokens)

    return tokenized_data


def get_words_with_nplus_frequency(tokenized_data, count_threshold):
    """
    Find words that appear at least count_threshold times in the data.

    Args:
        tokenized_data: List of lists of tokens.
        count_threshold: Minimum frequency for inclusion in vocabulary.

    Returns:
        vocabulary: A sorted list of words whose frequency >= count_threshold.
    """
    # Count word frequencies across all sentences
    word_counts = Counter()
    for sentence in tokenized_data:
        for word in sentence:
            word_counts[word] += 1

    # Keep only words meeting the minimum frequency
    vocabulary = sorted([word for word, count in word_counts.items()
                         if count >= count_threshold])
    return vocabulary


def replace_oov_words_by_unk(tokenized_data, vocabulary, unknown_token="<unk>"):
    """
    Replace out-of-vocabulary words with the unknown token.

    Args:
        tokenized_data: List of lists of tokens.
        vocabulary: List of vocabulary words (the closed vocabulary).
        unknown_token: Token to substitute for OOV words.

    Returns:
        replaced_data: List of lists with OOV words replaced by unknown_token.
    """
    vocab_set = set(vocabulary)  # O(1) lookup

    replaced_data = []
    for sentence in tokenized_data:
        replaced_sentence = [
            word if word in vocab_set else unknown_token
            for word in sentence
        ]
        replaced_data.append(replaced_sentence)

    return replaced_data


def preprocess_data(filename, count_threshold, special_tokens=None,
                    sample_delimiter='\n', split_ratio=0.8):
    """
    Preprocess data: tokenize, split into train/test, build vocabulary,
    and replace infrequent words with <unk>.

    Args:
        filename: Path to the corpus file.
        count_threshold: Words with count < this are treated as unknown.
        special_tokens: A SpecialTokens object (defaults created if None).
        sample_delimiter: Delimiter for splitting text into sentences.
        split_ratio: Fraction of data used for training.

    Returns:
        train_data_replaced: Training data with OOV words replaced.
        test_data_replaced: Test data with OOV words replaced.
        vocabulary: List of vocabulary words.
    """
    if special_tokens is None:
        special_tokens = SpecialTokens()

    # Tokenize the raw text into a list of sentence token-lists
    tokenized_data = read_and_tokenize_sentences(filename, sample_delimiter)

    # Train / test split
    train_size = int(len(tokenized_data) * split_ratio)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    # Build closed vocabulary from the training data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # Replace OOV words in both train and test sets
    train_data_replaced = replace_oov_words_by_unk(
        train_data, vocabulary, unknown_token=special_tokens.unknown_token)
    test_data_replaced = replace_oov_words_by_unk(
        test_data, vocabulary, unknown_token=special_tokens.unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary


# Question 2: N-Gram Counting
def count_n_grams(data, n, special_tokens=None):
    """
    Count all n-grams in the data.

    Each sentence is padded with (n-1) start tokens at the beginning
    and one end token at the end before n-gram extraction.

    Args:
        data: List of lists of words (tokenized sentences).
        n: Number of words in a sequence (the n-gram order).
        special_tokens: A SpecialTokens object with start, end, unk tokens.

    Returns:
        n_grams: A dictionary mapping each n-gram tuple to its frequency.
                 e.g. {('i', 'am'): 5, ('am', 'happy'): 3, ...}
    """
    if special_tokens is None:
        special_tokens = SpecialTokens()

    n_grams = {}

    for sentence in data:
        # Pad sentence: (n-1) start tokens + sentence + 1 end token
        padded = ([special_tokens.start_token] * (n - 1)
                  + sentence
                  + [special_tokens.end_token])

        # Slide a window of size n across the padded sentence
        for i in range(len(padded) - n + 1):
            n_gram = tuple(padded[i: i + n])
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

    return n_grams


# Question 3: Estimate the Probabilities
class NGramModel:
    """
    Container for n-gram model parameters.

    Stores n-gram and (n+1)-gram count dictionaries along with
    vocabulary size and smoothing parameter k.
    """

    def __init__(self, n_gram_counts, n_plus1_gram_counts, vocab_size, k=1.0):
        self.n_gram_counts = n_gram_counts           # dict: n-gram -> count
        self.n_plus1_gram_counts = n_plus1_gram_counts  # dict: (n+1)-gram -> count
        self.vocabulary_size = vocab_size             # |V|
        self.k = k                                    # smoothing parameter


def estimate_probabilities(context_tokens, ngram_model):
    """
    Estimate the probability of each possible next word given a context
    using add-k smoothing.

        P(w | context) = (C(context, w) + k) / (C(context) + k * |V|)

    Args:
        context_tokens: A tuple of words representing the prior n-gram context.
        ngram_model: An NGramModel object.

    Returns:
        probabilities: A dictionary mapping each candidate next word to
                       its estimated probability.
    """
    probabilities = {}
    context_tokens = tuple(context_tokens)

    k = ngram_model.k
    V = ngram_model.vocabulary_size

    # Count of the context (the n-gram prefix)
    context_count = ngram_model.n_gram_counts.get(context_tokens, 0)

    # Denominator for k-smoothing
    denominator = context_count + k * V

    # Scan all (n+1)-grams whose prefix matches the context
    for n_plus1_gram, count in ngram_model.n_plus1_gram_counts.items():
        if n_plus1_gram[:-1] == context_tokens:
            word = n_plus1_gram[-1]
            probabilities[word] = (count + k) / denominator

    return probabilities


# Question 4: Infer N-Grams (Predict Next Word)
def predict_next_word(sentence_beginning, model, special_tokens=None):
    """
    Predict the most likely next word given the beginning of a sentence.

    The input string is tokenized, and the last n tokens are used as context
    (padded with start tokens if necessary). The word with the maximum
    estimated probability is returned.

    Args:
        sentence_beginning: A string (the beginning of a sentence).
        model: An NGramModel object.
        special_tokens: A SpecialTokens object.

    Returns:
        next_word: The word most likely to appear next (string).
    """
    if special_tokens is None:
        special_tokens = SpecialTokens()

    # Tokenize the input
    tokens = nltk.word_tokenize(sentence_beginning.lower())

    # Determine context length n from the model's n-gram counts
    # (the keys of n_gram_counts have length n)
    sample_key = next(iter(model.n_gram_counts))
    n = len(sample_key)

    # Build context: take the last n tokens, pad with <s> if too short
    if len(tokens) < n:
        context = ([special_tokens.start_token] * (n - len(tokens))) + tokens
    else:
        context = tokens[-n:]

    context = tuple(context)

    # Estimate probabilities for all candidate next words
    probabilities = estimate_probabilities(context, model)

    # If no candidates found (completely unseen context), return <unk>
    if not probabilities:
        return special_tokens.unknown_token

    # Return the word with the highest probability
    next_word = max(probabilities, key=probabilities.get)
    return next_word


# Question 5: Extra Credit - Stylistic N-Grams
class StyleGram:
    """
    Stylistic N-Gram classifier and generator.

    Each author's full corpus is used to train a separate bigram model.
    A shared vocabulary across all authors ensures the smoothing denominators
    are on the same scale, enabling fair cross-author perplexity comparison.
    The author whose bigram model yields the highest average log-likelihood
    for the input passage is selected as the best style match.
    """

    def __init__(self, style_files):
        """
        Process all style files and train one bigram model per author.

        All preprocessing and training is completed before this
        constructor returns.

        Args:
            style_files: A list of filenames, one per author/style.
        """
        self.style_files = style_files
        self.models = []          # One NGramModel (bigram) per style
        self.vocabularies = []    # One vocabulary list per style

        special_tokens = SpecialTokens()

        # --- Phase 1: Read and tokenize all files ---
        all_tokenized = []
        for filepath in style_files:
            tokenized_data = read_and_tokenize_sentences(
                filepath, sample_delimiter='\n')
            all_tokenized.append(tokenized_data)

        # --- Phase 2: Build a shared vocabulary from all corpora ---
        # Using the union ensures that author-specific words (e.g. "thee",
        # "thou" for Shakespeare) are recognized rather than mapped to <unk>
        combined_data = []
        for td in all_tokenized:
            combined_data.extend(td)
        shared_vocabulary = get_words_with_nplus_frequency(
            combined_data, count_threshold=2)
        shared_vocab_size = len(shared_vocabulary)

        # --- Phase 3: Build per-author bigram models with shared vocab ---
        for tokenized_data in all_tokenized:
            # Replace OOV words using the shared vocabulary
            data_replaced = replace_oov_words_by_unk(
                tokenized_data, shared_vocabulary,
                unknown_token=special_tokens.unknown_token)

            # Build unigram and bigram counts from the full corpus
            unigram_counts = count_n_grams(data_replaced, 1, special_tokens)
            bigram_counts = count_n_grams(data_replaced, 2, special_tokens)

            model = NGramModel(
                n_gram_counts=unigram_counts,
                n_plus1_gram_counts=bigram_counts,
                vocab_size=shared_vocab_size,
                k=1.0
            )
            self.models.append(model)
            self.vocabularies.append(shared_vocabulary)

    def _compute_style_log_likelihood(self, tokens, style_idx):
        """
        Compute the average bigram log-likelihood of a token sequence
        under a given author's model.

        Because all models share the same vocabulary size, the smoothing
        denominators are directly comparable across authors.

        Args:
            tokens: List of OOV-replaced tokens (shared vocabulary).
            style_idx: Index of the style to evaluate.

        Returns:
            avg_log_likelihood: Average log-probability per bigram.
        """
        model = self.models[style_idx]
        k = model.k
        V = model.vocabulary_size
        special_tokens = SpecialTokens()

        # Pad with start and end tokens, same as during training
        padded = [special_tokens.start_token] + tokens + [special_tokens.end_token]
        num_bigrams = len(padded) - 1
        log_likelihood = 0.0

        for i in range(num_bigrams):
            context_count = model.n_gram_counts.get((padded[i],), 0)
            bigram_count = model.n_plus1_gram_counts.get(
                (padded[i], padded[i + 1]), 0)
            # Add-k smoothed probability
            prob = (bigram_count + k) / (context_count + k * V)
            log_likelihood += math.log(prob)

        # Normalize by number of bigrams to avoid length bias
        if num_bigrams > 0:
            log_likelihood /= num_bigrams

        return log_likelihood

    def write_in_style_ngram(self, passage):
        """
        Identify the most likely author style for a passage and predict
        the next word using that author's bigram model.

        Args:
            passage: A string containing a passage.

        Returns:
            word: The single best predicted next word (string).
            probability_word: Probability of the predicted word (float).
            style_index: Index of the matched style file (int).
            probability_style: Probability of the matched style (float).
            top_10_words: List of (word, probability) tuples for the ten
                          most likely next words in the matched style.
        """
        special_tokens = SpecialTokens()

        # Tokenize the passage
        tokens = nltk.word_tokenize(passage.lower())

        # Replace OOV words using the shared vocabulary
        vocab_set = set(self.vocabularies[0])  # shared across all styles
        tokens_replaced = [
            t if t in vocab_set else special_tokens.unknown_token
            for t in tokens
        ]

        # --- Step 1: Classify the style ---
        # Score passage under each author's bigram model
        log_likelihoods = []
        for i in range(len(self.style_files)):
            ll = self._compute_style_log_likelihood(tokens_replaced, i)
            log_likelihoods.append(ll)

        # Softmax to convert log-likelihoods into style probabilities
        max_ll = max(log_likelihoods)
        exp_values = [math.exp(ll - max_ll) for ll in log_likelihoods]
        total_exp = sum(exp_values)
        style_probs = [ev / total_exp for ev in exp_values]

        # Select the style with the highest probability
        best_style_idx = style_probs.index(max(style_probs))
        probability_style = style_probs[best_style_idx]

        # --- Step 2: Predict next word with the best style's bigram model ---
        model = self.models[best_style_idx]

        # Context for bigram model: last token (unigram context)
        if len(tokens_replaced) > 0:
            context = (tokens_replaced[-1],)
        else:
            context = (special_tokens.start_token,)

        # Estimate next-word probabilities
        probabilities = estimate_probabilities(context, model)

        if not probabilities:
            return (special_tokens.unknown_token, 0.0,
                    best_style_idx, probability_style, [])

        # Sort by probability descending and take top 10
        sorted_probs = sorted(probabilities.items(),
                              key=lambda x: x[1], reverse=True)
        top_10_words = sorted_probs[:10]

        # The single best word
        best_word = top_10_words[0][0]
        probability_word = top_10_words[0][1]

        return best_word, probability_word, best_style_idx, probability_style, top_10_words



# Main: Build models and demonstrate predictions
if __name__ == "__main__":
    print("=" * 70)
    print("CS 6120 - Assignment 4: N-Gram Language Model")
    print("=" * 70)

    special_tokens = SpecialTokens()


    # Q1: Preprocessing
    print("\n--- Q1: Preprocessing ---")
    train_data, test_data, vocabulary = preprocess_data(
        "as4_file/en_US.twitter.txt", count_threshold=2, special_tokens=special_tokens)

    print(f"Training sentences : {len(train_data)}")
    print(f"Test sentences     : {len(test_data)}")
    print(f"Vocabulary size    : {len(vocabulary)}")
    print(f"Sample train[0]    : {train_data[0][:10]}")


    # Q2: N-Gram Counting  (small verification + full corpus)
    print("\n--- Q2: N-Gram Counting ---")

    # Verify on the small example from the assignment spec
    sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
    bigram_counts_small = count_n_grams(sentences, 2, SpecialTokens())
    print("Small-example bigram counts (verification):")
    for gram in [('<s>', 'i'), ('<s>', 'this'), ('a', 'cat'),
                 ('like', 'a'), ('cat', '<e>')]:
        print(f"  {gram}: {bigram_counts_small.get(gram, 0)}")

    # Full corpus counts
    unigram_counts = count_n_grams(train_data, 1, special_tokens)
    bigram_counts  = count_n_grams(train_data, 2, special_tokens)
    trigram_counts = count_n_grams(train_data, 3, special_tokens)
    print(f"\nFull corpus counts:")
    print(f"  Unique unigrams : {len(unigram_counts)}")
    print(f"  Unique bigrams  : {len(bigram_counts)}")
    print(f"  Unique trigrams : {len(trigram_counts)}")


    # Q3: Probability Estimation
    print("\n--- Q3: Probability Estimation ---")

    # Bigram model: context = unigram, predict next word from bigrams
    bigram_model = NGramModel(
        n_gram_counts=unigram_counts,
        n_plus1_gram_counts=bigram_counts,
        vocab_size=len(vocabulary),
        k=1.0
    )
    # Trigram model: context = bigram, predict next word from trigrams
    trigram_model = NGramModel(
        n_gram_counts=bigram_counts,
        n_plus1_gram_counts=trigram_counts,
        vocab_size=len(vocabulary),
        k=1.0
    )

    # Demo: top-5 words after "i" (bigram)
    probs = estimate_probabilities(("i",), bigram_model)
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 words after 'i' (bigram model):")
    for w, p in top5:
        print(f"  {w}: {p:.6f}")

    # Demo: top-5 words after "i am" (trigram)
    probs_tri = estimate_probabilities(("i", "am"), trigram_model)
    top5_tri = sorted(probs_tri.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 words after 'i am' (trigram model):")
    for w, p in top5_tri:
        print(f"  {w}: {p:.6f}")


    # Q4: Predict Next Word
    print("\n--- Q4: Predict Next Word ---")

    test_sentences = [
        "I would like to",
        "The next word is",
        "Hey how are",
        "I am so",
        "Today I want to",
    ]

    print("Bigram model predictions:")
    for s in test_sentences:
        w = predict_next_word(s, bigram_model, special_tokens)
        print(f"  '{s}' -> '{w}'")

    print("Trigram model predictions:")
    for s in test_sentences:
        w = predict_next_word(s, trigram_model, special_tokens)
        print(f"  '{s}' -> '{w}'")


    # Q5: Extra Credit - Stylistic N-Grams
    print("\n--- Q5: Stylistic N-Grams (Extra Credit) ---")

    style_files = [
        "as4_file/hemingway-edit.txt",
        "as4_file/pg100.txt",       # Shakespeare
        "as4_file/pg12242.txt"      # Dickinson
    ]
    author_names = ["Hemingway", "Shakespeare", "Dickinson"]

    print("Training style models (this may take a moment)...")
    stylegram = StyleGram(style_files)
    print("Done.\n")

    test_passages = [
        "The old man sat by the river and watched the boats",
        "To be or not to be that is the question whether",
        "The soul selects her own society then shuts the door",
        "I love going to the beach with my friends",
    ]

    for passage in test_passages:
        word, prob_w, idx, prob_s, top_10 = stylegram.write_in_style_ngram(passage)
        print(f"  Passage : '{passage}'")
        print(f"    Style : {author_names[idx]} (prob = {prob_s:.4f})")
        print(f"    Top 10 next words in {author_names[idx]} style:")
        for rank, (w, p) in enumerate(top_10, 1):
            print(f"      {rank:2d}. {w:15s} (prob = {p:.6f})")
        print()

    print("Assignment 4 complete.")