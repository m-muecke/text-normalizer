import unicodedata
from types import GeneratorType
from typing import List, Optional, Set

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class TextNormalizerNLTK(TransformerMixin, BaseEstimator):
    """
    Attributes:
        stopwords (set): NLTK english stopwords. 
        lemmatizer (WordNetLemmatizer): Instance of WordNetLemmatizer.

    Args:
        remove_stopwords: Remove stopwords in transform method or don't.
        apply_lemmatizer: Lemmatize tokens in transform method or don't.
    """
    def __init__(self,
                 remove_stopwords: Optional[bool] = True,
                 apply_lemmatize: Optional[bool] = True):
        self.stopwords  = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.remove_stopwords = remove_stopwords
        self.apply_lemmatize = apply_lemmatize

    def add_stopwords(self, stopwords: Set[str]):
        """Add stopwords to stopwords set."""
        self.stopwords = self.stopwords.update(stopwords)

    def replace_stopwords(self, stopwords: Set[str]):
        """Replace stopwords with own stopwords."""
        self.stopwords = stopwords

    def is_punct(self, token: str) -> bool:
        """Checks if token is punctuation or not."""
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token: str) -> bool:
        """Checks if token is in stopword list or not."""
        return token.lower() in self.stopwords

    def lemmatize(self, token: str) -> str:
        """Lemmatizes given input token."""
        pos_tag = nltk.pos_tag(token)[0][1]
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self,
                  text: str,
                  remove_stopwords: Optional[bool] = True,
                  apply_lemmatize: Optional[bool] = True) -> List[str]:
        """Tokenize and normalize text. 
        
        Args:
            text: Input text.
            remove_stopwords: Should stopwords be removed or not.
            apply_lemmatize: Should lemmatization occur or not.
        
        Returns:
            Normalized string list.
        """
        if remove_stopwords and apply_lemmatize:
            return [
                self.lemmatize(token)
                for token in nltk.word_tokenize(text)
                if not self.is_punct(token) and not self.is_stopword(token)
            ]
        elif remove_stopwords and not apply_lemmatize:
            return [
                token
                for token in nltk.word_tokenize(text)
                if not self.is_punct(token) and not self.is_stopword(token)
            ]
        elif not remove_stopwords and apply_lemmatize:
            return [
                self.lemmatize(token)
                for token in nltk.word_tokenize(text)
                if not self.is_punct(token)
            ]
        return [
            token
            for token in nltk.word_tokenize(text)
            if not self.is_punct(token)
        ]
    
    def fit(self, X, y: Optional = None):
        return self

    def transform(self, documents: List[str]) -> GeneratorType:
        for document in documents:
            yield self.normalize(document,
                                 remove_stopwords=self.remove_stopwords,
                                 apply_lemmatize=self.apply_lemmatize)

if __name__ == '__main__':

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
        "Isn't this great?",
        "Wouldn't this be great?",
        'Is this great, walking in the park?',
        'She talks on the phone.',
    ]
 
    text_normalizer = TextNormalizerNLTK()
    normalized_corpus = list(text_normalizer.transform(corpus))
