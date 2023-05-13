from typing import Union

from tiblib import ClassifierBase, TransformerBase


class Pipeline(ClassifierBase):
    '''
    Pipeline class containing n transformers and one final classifier.
    Used to standardize preprocessing and avoid data leakage
    '''
    def __init__(self, transformers: Union[list, TransformerBase] , classifier: ClassifierBase):
        if len(transformers) == 1 and type(transformers) is not list:
            self.transformers = [transformers]
        else:
            self.transformers = transformers
        self.classifier = classifier

    def __str__(self):
        return str(self.classifier)

    def fit(self, X, y):
        X_transf = X
        for t in self.transformers:
            X_transf = t.fit_transform(X_transf)
        self.classifier.fit(X_transf,y)
        return self

    def predict(self, X):
        X_transf = X
        for t in self.transformers:
            X_transf = t.transform(X_transf)
        return self.classifier.predict(X_transf)

    def predict_scores(self, X, get_ratio=False):
        X_transf = X
        for t in self.transformers:
            X_transf = t.transform(X_transf)
        return self.classifier.predict_scores(X_transf, get_ratio=get_ratio)
