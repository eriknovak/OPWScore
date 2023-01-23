import math
import pickle

# used for documentation
from typing import List


class WeightStore:
    """The class for storing vocabulary
    The vocabulary class is used to index the words of a particular
    dataset and for storing some basic word statistics.
    Attributes:
        word2index: The word-to-index dictionary.
        index2word: The index-to-word dictionary.
        word_count: The dictionary containing the word's frequency.
        n_words: The number of unique words in the vocabulary.
        word_document_count: The dictionary containing the word's document count.
        n_documents: The number of documents.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        """Initializes the vocabulary instance"""
        self.word2index = {}
        self.index2word = {}
        self.word_count = {}
        self.n_words = 0
        self.word_doc_count = {}
        self.n_documents = 0
        self.eps = eps

    def add_word(self, word: str, token_id: int) -> None:
        """Add a word to the vocabulary.
        Args:
            word: The added word.
        """
        if word in self.word_count:
            self.word_count[word] += 1
        else:
            self.word2index[word] = token_id
            self.index2word[token_id] = word
            self.word_count[word] = 1
            self.n_words += 1

    def add_word_document(self, word: str) -> None:
        """Update the word-to-document count statistics.
        Args:
            word: The word to update statistics.
        """
        if word in self.word_doc_count:
            self.word_doc_count[word] += 1
        else:
            self.word_doc_count[word] = 1

    def add_document(self, document: str, tokenizer) -> None:
        """Adds the document's terms to the vocabulary
        Args:
            document: The added document.
        """
        # iterate through the document terms
        self.n_documents += 1
        document_terms = set()  # store the unique document terms
        token_ids = tokenizer(document, padding=False, truncation=False)["input_ids"]
        terms = tokenizer.convert_ids_to_tokens(token_ids)
        for term, token_id in zip(terms, token_ids):
            # update the term statistics
            self.add_word(term, token_id)
            document_terms.add(term)
        # update the word-to-document statistics
        for term in document_terms:
            self.add_word_document(term)

    def add_corpus(self, corpus: List[str], tokenizer) -> None:
        """Adds the corpus of documents to the vocabulary
        Args:
            corpus: The list of documents.
        """
        for document in corpus:
            self.add_document(document, tokenizer)

    def get_word(self, token_id: int) -> str:
        """Gets the word based associated with the given token ID
        Args:
            token_id: The token id.
        Returns:
            The word associated with the token id.
        """
        return self.index2word[token_id] if token_id in self.index2word else "[UNK]"

    def get_idf(self, word: str) -> float:
        """Gets the idf value of the given word
        Calculates the inverse document frequency (IDF)
        of the given word. It is calculated as `-log(n/N)`,
        where `N` is the total number of documents the
        vocabulary has seen and `n` is the number of
        documents containing the word.
        Args:
            word: The word string.
        Returns:
            The float number representing the idf score
            of the term.
        """
        word_doc_freq = self.word_doc_count[word] if word in self.word_doc_count else 0
        return -math.log((1 + word_doc_freq) / (1 + self.n_documents)) + self.eps

    def get_ft(self, word: str) -> float:
        """Gets the ft value of the given word
        Retrieves the term frequency (TF) value of the
        given word.
        Args:
            word: The word string.
        Returns:
            The float number representing the ft score
            of the term.
        """
        return self.word_count[word] if word in self.word_count else 0

    def save(self, filename: str) -> None:
        """Save the weight store in a file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename: str):
        """Load the weight store from a file"""
        with open(filename, "rb") as f:
            return pickle.load(f)
