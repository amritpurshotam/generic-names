import sys
sys.path.append('src')

from data import read_raw_data, clean, vectorize
from xgboost import XGBClassifier
from visualisation import plot_precision_recall_vs_threshold
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == "__main__":
    df = read_raw_data('data\\raw\\A_training data.csv')
    df = clean(df)
    features, vectorizer = vectorize(df.name.values)
    labels = df.name_generic.values

    params = {
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.5), # default 0.1 
        "max_depth": randint(2, 10), # default 3
        "n_estimators": randint(100, 300), # default 100
        "subsample": uniform(0.6, 0.4),
        "scale_pos_weight": randint(5, 40)
    }

    model = XGBClassifier()
    search = RandomizedSearchCV(model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=4, return_train_score=True, scoring='f1')

    search.fit(features, labels)

    report_best_scores(search.cv_results_, 1)
