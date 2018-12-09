import sys
sys.path.append('src')

from data import read_raw_data, clean, vectorize
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from visualisation import plot_precision_recall_vs_threshold
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    df = read_raw_data('data\\raw\\A_training data.csv')
    df = clean(df)
    features = df.name.values
    labels = df.name_generic.values

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, stratify=labels, test_size=0.2)
    train_features, vectorizer = vectorize(train_features)
    test_features = vectorizer.transform(test_features)

    weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    clf = XGBClassifier(scale_pos_weight=weight, max_depth=9, n_estimators=300, learning_rate=0.35, gamma=0.08, subsample=0.7)
    clf.fit(train_features, train_labels)

    predictions = clf.predict_proba(test_features)

    plot_precision_recall_vs_threshold(test_labels, predictions)