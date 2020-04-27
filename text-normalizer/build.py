from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
# local import
from normalize import TextNormalizerNLTK

def identity(words):
    """Identity function for tokenizer param in CountVectorizer."""
    return words

def create_pipeline(estimator=None):
    """Create sklearn Pipeline object."""
    steps = [
        ('norm', TextNormalizerNLTK()),
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
    pass
