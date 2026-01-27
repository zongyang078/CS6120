# Shakespeare Autocomplete

A fast, Trie-based word autocomplete system trained on the complete works of Shakespeare. Features a web interface built with Streamlit for interactive exploration.

## Overview

This project implements an efficient autocomplete engine that suggests the most frequently used words from Shakespeare's corpus based on a given prefix. The system uses a Trie data structure with pre-computed top-10 caches at each node, enabling sub-millisecond query times.

**Example**: Typing `th` returns `['the', 'that', 'this', 'thou', 'thy', 'thee', 'they', 'then', 'there', 'their']`

## Features

- **Fast Lookups**: O(m) query time where m is the prefix length
- **Frequency-Based Suggestions**: Returns the top 10 most common words matching the prefix
- **Interactive Web UI**: Real-time suggestions with query timing display
- **Comprehensive Corpus**: Trained on 121,000+ lines of Shakespeare's works

## Project Structure

```
CS6120/
├── app.py              # Streamlit web application
├── assignment1.py      # Core autocomplete implementation
├── main.py             # Data download utility
├── shakespeare-edit.txt # Shakespeare corpus (5.1 MB)
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CS6120
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install streamlit
   ```

4. Download the corpus (if not included):
   ```bash
   python main.py
   ```

## Usage

### Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` where you can type prefixes and see autocomplete suggestions in real-time.

### Command Line

Run the module directly to execute the test suite:

```bash
python assignment1.py
```

### Programmatic Usage

```python
from assignment1 import read_vocabulary, process_data, autocomplete_word

# Load and process the corpus
vocabulary = read_vocabulary("shakespeare-edit.txt")
trie = process_data(vocabulary)

# Get suggestions for a prefix
suggestions = autocomplete_word(trie, "lov")
print(suggestions)  # ['love', 'lord', 'lovers', 'loved', 'loving', ...]
```

## Implementation Details

### Architecture

1. **Vocabulary Processing** (`read_vocabulary`): Reads the corpus, removes punctuation/numbers, normalizes case, and returns unique words sorted by frequency (most common first).

2. **Trie Construction** (`process_data`): Builds a Trie where each node maintains a cache of the top 10 most frequent words passing through it.

3. **Query Execution** (`autocomplete_word`): Traverses the Trie to the prefix endpoint and returns the pre-computed top-10 suggestions.

### Performance

- **Preprocessing**: One-time O(n log n) for sorting + O(n × w) for Trie construction
- **Query Time**: O(m) where m = prefix length, typically < 2ms
- **Memory**: Proportional to vocabulary size with top-10 caches at each node

## API Reference

### `read_vocabulary(file_path: str) -> list[str]`

Reads and preprocesses a text file, returning unique words sorted by frequency.

### `process_data(vocabulary: list[str]) -> TrieNode`

Builds a Trie data structure from the vocabulary with top-10 caching.

### `autocomplete_word(root: TrieNode, prefix: str) -> list[str]`

Returns up to 10 word suggestions for the given prefix, or an empty list if no matches exist.

## Testing

The module includes 8 test cases covering:

- Common prefixes (`th`, `lo`, `wh`)
- Single-character prefixes (`a`, `b`)
- Empty prefix (returns overall top 10)
- Non-existent prefixes (returns empty list)
- Case normalization

Run tests with:

```bash
python assignment1.py
```

## License

This project was created for CS6120 coursework.
