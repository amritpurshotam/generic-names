import sys
sys.path.append('src')
from data import clean, vectorize
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from numpy import ndarray
import pickle

class XgbModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, df: DataFrame) -> None:
        df = clean(df)
        features, self.vectorizer = vectorize(df.name.values)
        labels = df.name_generic.values

        self.model = XGBClassifier(scale_pos_weight=32, max_depth=9, n_estimators=300, learning_rate=0.35, gamma=0.08, subsample=0.7)
        self.model.fit(features, labels)

    def predict(self, raw_names: ndarray) -> ndarray:
        features = self.vectorizer.transform(raw_names)
        predictions = self.model.predict_proba(features)
        predictions = predictions[:,1:]
        predictions[predictions > 0.2] = 1
        predictions[predictions <= 0.2] = 0
        return predictions

    def load(self, dir: str):
        with open(dir + '/model.pickle', 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(dir + '/vectorizer.pickle', 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
    
    def save(self, dir: str):
        with open(dir + '/model.pickle', 'wb') as model_file:
            pickle.dump(file=model_file, obj=self.model)
        with open(dir + '/vectorizer.pickle', 'wb') as vectorizer_file:
            pickle.dump(file=vectorizer_file, obj=self.vectorizer)