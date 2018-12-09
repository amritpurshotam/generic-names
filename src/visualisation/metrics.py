from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from numpy import ndarray

def plot_precision_recall_vs_threshold(labels: ndarray, predictions: ndarray) -> None:
    precisions, recalls, thresholds = precision_recall_curve(labels, predictions[:,1:])

    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.ylabel('Score')
    plt.xlabel('Decision Threshold')
    plt.legend(loc='best')
    plt.show()