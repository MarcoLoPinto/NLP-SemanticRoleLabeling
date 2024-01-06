from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary

def print_classification_report(y_true, y_pred, target_names):
    print( classification_report(y_true, y_pred, target_names=target_names) )


def plot_confusion_matrix(y_true, y_pred, display_labels):
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(display_labels))], normalize='pred') # normalized over predicted label: the sum of each column is 1
    cmd = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(22,22))
    cmd.plot(ax=ax)
    plt.xticks(rotation=90)

def print_summary(model):
    print(model)
    print('----------------------')
    p = sum(p.numel() for p in model.parameters())
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntp = p - tp
    print('parameters:', f'{p:,}')
    print('trainable parameters:', f'{tp:,}')
    print('non-trainable parameters:', f'{ntp:,}')

def display_history(dict_history):
    plt.figure(figsize=(8,8))
    for name, hist in dict_history.items():
        plt.plot([i for i in range(len(hist))], hist, label=name)
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('Model learning')
    plt.legend()
    plt.show()