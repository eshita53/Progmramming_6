import pandas as pd
import numpy as np
from helping_function import setup_logging
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os
import pickle


def main():

    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    model_name = snakemake.params.model_name

    with open(snakemake.input.model, "rb") as f:
        loaded_model = pickle.load(f)

    logger.info("Reading the data..")
    X_test = pd.read_csv(snakemake.input.test_X)
    y_pred = loaded_model.predict(X_test)
    y_true = pd.read_csv(snakemake.input.test_y)
    logger.info("Data retival has been done.")
    classes = np.unique(y_true)
    # converting y_true to one hot encoding. As we need one-hot encoding to calculate roc_curve for multiclass
    y_true_one_hot = label_binarize(y_true, classes=np.arange(len(classes)))
    logger.info(
        "Calculating Model Evalution metrics like accuracy, f1 score, roc-curve.")
    accuracy = accuracy_score(y_true, y_pred)
    predict_proba = loaded_model.predict_proba(X_test)
    f1 = f1_score(y_true, y_pred, average='micro')
    # as we have multiclass outcome and class is imbalance
    # That's why we will calculate multiclass roc curve with micro-average ROC
    fpr, tpr, roc_auc = {}, {}, {}
    # Computing ROC for each class
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(
            y_true_one_hot[:, i], predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(
        y_true_one_hot.flatten(), predict_proba.flatten())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    metrics_dict = {
        'model_name': model_name,
        'accuracy': accuracy,
        "f1_score": f1,
        "auc_micro": roc_auc['micro'],
    }
    # saving the class specific values
    for i in range(len(classes)):
        metrics_dict[f"auc_class_{classes[i]}"] = roc_auc[i]
        # converting the arrays to json string because we can not directly store array values in ai=single csv cell
        metrics_dict[f"fpr_class_{classes[i]}"] = json.dumps(fpr[i].tolist())
        metrics_dict[f"tpr_class_{classes[i]}"] = json.dumps(tpr[i].tolist())

    metrics_dict['fpr_micro'] = json.dumps(fpr['micro'].tolist())
    metrics_dict['tpr_micro'] = json.dumps(tpr['micro'].tolist())

    # as out dict has multiple array values that's why creating single-row DataFrame with all the metrics
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(snakemake.output.metrics, index=False)
    logger.info("Model evaluation metrics has been saved  to a csv file!")


if __name__ == "__main__":
    main()
