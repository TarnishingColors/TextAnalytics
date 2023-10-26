from functools import reduce
from random import choice
from typing import List, Union, Dict, Set, Tuple, Optional, Deque, Sequence
import numpy as np
from nltk import ngrams
from numpy.typing import NDArray
from collections import defaultdict, deque, Counter
from nltk.corpus.reader.util import StreamBackedCorpusView
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class BaseTextCorpus:
    def _get_valid_tokens(self, corpus: Union[List[str], StreamBackedCorpusView], remove_stopwords: bool = False) -> List[str]:
        """
        :param corpus: list of tokens that should be cleaned
        :param remove_stopwords: bool indicating if stopwords should be removed
                                 False by default
        :return: list of valid tokens
        """
        stop_words = set(stopwords.words('english')) if remove_stopwords else {}
        integer_pattern = r'\b\d+\b'
        valid_tokens = [
            token.lower() for token in corpus
            if token not in string.punctuation
               and not re.search(integer_pattern, token)
               and (True if not remove_stopwords else token not in stop_words)
        ]
        return valid_tokens


class WordSimilarity(BaseTextCorpus):
    def __init__(self, corpus: Union[List[str], StreamBackedCorpusView], input_word: str, context_size: Optional[int] = 3):
        """
        :param corpus: list of tokens
        :param input_word: string that represents the word we are interested in
        :param context_size: integer that indicates the number of context words that are considered on both sides of the central word
        """
        self.tokens = self._get_valid_tokens(corpus, remove_stopwords=True)
        self.context_size = context_size
        self.input_word = input_word

    def find_most_similar_words(self) -> Tuple[Set[str], float]:
        """
        :returns:
            - set of the most similar words to the input word
            - float that indicates the highest Jaccard similarity to the input word
        """
        return self._most_similar_words(self._to_sets(self._get_surrounding_counts()))

    def _get_surrounding_counts(self) -> Dict[str, Dict[str, int]]:
        """
        :return: dict of dicts that holds the count of context words for each input token
        """
        result = {}
        size = len(self.tokens)
        for i, token in enumerate(self.tokens):
            if token not in result:
                result[token] = defaultdict(int)
            for j in range(max(0, i - self.context_size), min(size, i + self.context_size + 1)):
                if j != i:
                    result[token][self.tokens[j]] += 1
        return result

    def _to_sets(self, context_dict: Dict[str, Dict[str, int]], k: Optional[int] = 20) -> Dict[str, Set[str]]:
        """
        :param context_dict: dict of dicts that holds the count of context words for each word
        :param k: integer that specifies how many context words should be kept
        :return: dict that maps each word to a set of its k most frequent context words
        """
        result = {}
        for key, value in context_dict.items():
            sorted_items = sorted(value.items(), key=lambda x: (-x[1], x[0]))[:k]
            result[key] = list(dict(sorted_items).keys())
        return result

    def _most_similar_words(self, contexts: Dict[str, Set[str]]) -> Tuple[Set[str], float]:
        """
        :param contexts: dict that maps each word to a set of its most frequent context words
        :returns:
            - set of the most similar words to the input word
            - float that indicates the highest Jaccard similarity to the input word
        """
        max_jaccard_similarity = -1
        closest_words = set()
        word_closest = set(contexts[self.input_word])
        for key, value in contexts.items():
            if key != self.input_word:
                current_jaccard_similarity = len(word_closest.intersection(value)) / len(word_closest.union(value))
                if current_jaccard_similarity > max_jaccard_similarity:
                    max_jaccard_similarity = current_jaccard_similarity
                    closest_words = {key}
                elif current_jaccard_similarity == max_jaccard_similarity:
                    closest_words.add(key)

        return closest_words, max_jaccard_similarity


class MinimumCostAlignment:
    def __init__(self, word1: str, word2: str, cost_of_substitute: Optional[int] = 2):
        self.word1 = word1
        self.word2 = word2
        self.cost_of_substitute = cost_of_substitute
        self.D = self._edit_distance()

    def get_alignment(self) -> Tuple[str, str]:
        """
        :returns: tuple of strings that indicate the alignment of the input strings
        """
        def find_previous(x: int, y: int) -> Tuple[int, int, str]:
            if x != 0 and y != 0:
                if self.D[x - 1][y - 1] - self.D[x][y] == -2:
                    return x - 1, y - 1, 's'
            if x != 0 and self.D[x - 1][y] == self.D[x][y] - 1:
                return x - 1, y, 'd'
            if y != 0 and self.D[x][y - 1] == self.D[x][y] - 1:
                return x, y - 1, 'i'
            else:
                return x - 1, y - 1, 's'

        def retrieve_alignments(operations: Deque[str], str1: str, str2: str) -> Tuple[str, str]:
            alignment_str1 = alignment_str2 = ''
            i = j = 0
            for operation in operations:
                if operation == 'i':
                    alignment_str1 += "*"
                    alignment_str2 += str2[j]
                    j += 1
                elif operation == 'd':
                    alignment_str1 += str1[i]
                    alignment_str2 += "*"
                    i += 1
                else:
                    alignment_str1 += str1[i]
                    alignment_str2 += str2[j]
                    i += 1
                    j += 1
            return alignment_str1, alignment_str2

        x, y = len(self.D) - 1, len(self.D[0]) - 1

        operations = deque()

        while x > 0 or y > 0:
            x, y, move_type = find_previous(x, y)
            operations.appendleft(move_type)

        result_alignments = retrieve_alignments(operations, self.word1, self.word2)

        return result_alignments

    def _edit_distance(self) -> Tuple[int, NDArray[NDArray[int]]]:
        """
        :returns:
            - minimum edit distance table
        """
        I = len(self.word1) + 1
        J = len(self.word2) + 1
        D = np.zeros((I, J), dtype=int)

        D[:, 0] = np.arange(I)
        D[0, :] = np.arange(J)

        for i in range(1, I):
            for j in range(1, J):
                sub = D[i-1, j-1]
                if self.word1[i-1] != self.word2[j-1]:
                    sub += self.cost_of_substitute

                D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, sub)

        return D


class MarkovModel(BaseTextCorpus):
    """Markov model for generating text."""

    def __init__(self, corpus: Union[List[str], StreamBackedCorpusView], context_len: int):
        """
        :param tokens: text corpus on which the model is trained on as an iterator of tokens
        :param context_len: length of the n-gram (number of preceding words)
        """
        self.context_len = context_len
        self.counts, self.v = self._process_corpus(self._get_valid_tokens(corpus))

    def _process_corpus(self, tokens: Sequence[str]) -> Tuple[Dict[Tuple[str, ...], Dict[str, int]], Set]:
        """
        Training method of the model, counts the occurences of each word after each observed n-gram.
        :param tokens: text corpus on which the model is trained on as an iterator of tokens
        :returns:
            - nested dict that maps each n-gram to the counts of the words succeeding it
            - the whole vocabulary as a set
        """
        n_grams = list(ngrams(tokens, self.context_len + 1))
        counts = {}
        for key, value in Counter(n_grams).items():
            if key[:-1] in counts:
                counts[key[:-1]].update({key[-1]: value})
            else:
                counts[key[:-1]] = {key[-1]: value}
        return counts, set(tokens)

    def transition_prob(self, ngram: Tuple[str, ...], word: str) -> float:
        """
        Compute the conditional probability that the input word follows the given n-gram.
        :param ngram: string tuple that represents an n-gram
        :param word: input word
        :return: probability that the n-gram is followed by the input word
        """
        return self.counts[ngram][word] / reduce(lambda acc, val: acc + val, self.counts[ngram].values()) if word in self.counts[ngram] else 1 / len(self.v)

    def most_likely_word(self, ngram: Tuple[str, ...]) -> Set[str]:
        """
        Computes which word is most likely to follow a given n-gram.
        :param ngram: n-gram we are interested in
        return: set of words that are most likely to follow the n-gram
        """
        if ngram not in self.counts:
            return self.v
        highest = max(self.counts[ngram].values())
        return set(key for key, value in self.counts[ngram].items() if value == highest)

    def generate_text(self, ngram: Tuple[str, ...], k: int) -> List[str]:
        """
        Generates a text sequence of length k, given a starting sequence.
        :param ngram: starting sequence
        :param k: total number of words in the generated sequence
        :return: sequence of generated words, including the starting sequence
        """
        continuation = []
        context = deque(ngram, maxlen=self.context_len)
        for i in range(k):
            next_options = self.most_likely_word(tuple(context))
            next_word = next_options.pop() if len(next_options) == 1 else choice(tuple(next_options))
            continuation.append(next_word)
            context.append(next_word)
        return continuation
