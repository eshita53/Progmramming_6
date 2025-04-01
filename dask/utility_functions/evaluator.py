from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, auc, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

class Evaluator:
    def __init__(self, model,):
        pass
        
    @staticmethod
    def eval(test_y, y_pred, print_result = False):
        
        accuracy = accuracy_score(test_y, y_pred)

        if print_result:
            report = classification_report(test_y, y_pred, zero_division=0)
            matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_pred, test_y))
            print("Model testing complete.")
            print(f"Accuracy: {accuracy}")
            print("\nClassification Report:\n", report)
            print("\nMatrix:\n")
            matrix.plot()

        return accuracy
    
    @staticmethod
    def sklearn_roc_curve(test_y,predict_proba):
        
        # fpr, tpr, thresholds = roc_curve(test_y, predict_proba)
        auc_score = roc_auc_score(test_y, predict_proba, multi_class='ovo', average='macro')
        return auc_score


    @staticmethod
    def save_the_roc_curve(fpr,tpr,roc_auc, file_path):
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--') 
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(file_path)

    @staticmethod
    def f1_score_recall_precision_calculation(test_y, y_pred):
        report = classification_report(test_y, y_pred, output_dict=True, zero_division=0)
        report =report['macro avg']
        f1_score_macro_avg = report['f1-score']
        precision_macro_avg = report['precision']
        recall_macro_avg = report['recall']
        
        return f1_score_macro_avg, precision_macro_avg, recall_macro_avg



    @staticmethod
    def plot_roc_curve(y_true, predict_proba, save_path):
        
        classes = np.unique(y_true)
        y_true_one_hot = label_binarize(y_true, classes=np.arange(len(classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        # Computing ROC for each class
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(
                y_true_one_hot[:, i], predict_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr['micro'], tpr['micro'], _ = roc_curve(
            y_true_one_hot.flatten(), predict_proba.flatten())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        Evaluator.save_the_roc_curve(fpr['micro'], tpr['micro'], roc_auc['micro'], save_path)