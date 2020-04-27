import nltk
import unicodedata
from types import GeneratorType
from typing import List, Optional, Set

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

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
        """"""
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
        """"""
        return self

    def transform(self, documents: List[str]) -> GeneratorType:
        """"""
        for document in documents:
            yield self.normalize(document,
                                 remove_stopwords=self.remove_stopwords,
                                 apply_lemmatize=self.apply_lemmatize)

def identity(words):
    """Identity function for tokenizer param in CountVectorizer."""
    return words

def create_pipeline(estimator=None):
    """Create sklearn Pipeline object."""
    steps = [
        ('norm', TextNormalizer()),
        ('vect', CountVectorizer(
            tokenizer=identity, lowercase=False
        )),
        ('tfidf', TfidfTransformer())
    ]
    # return w/o estimator
    if not estimator:
        return Pipeline(steps)
    # add the estimator
    steps.append(('clf', estimator))
    return Pipeline(steps)
    
def run_multiclass():
    """Add multiclass estimators to Pipeline."""
    model_list = []
    for form in (MultinomialNB, GaussianNB, 
                 LogisticRegression, SGDClassifier):
        model_list.append(create_pipeline(form()))
    return model_list

if __name__ == '__main__':
    # TODO: add ML functionalities
    # TODO: add docstrings

    import pandas as pd

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
    labels = [1, 1, 1, 2, 2, 2, 3]
    s_corpus = pd.Series(corpus)
    
    text_normalizer = TextNormalizer()
    
    create_pipeline().fit
    print('input as list')
    print(create_pipeline().fit_transform(corpus))

    print(run_multiclass())
    # GridsearchCV: https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
