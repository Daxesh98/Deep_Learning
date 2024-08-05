from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model (y_test, y_preds):
    accuracy = accuracy_score(y_test, y_preds)
    return accuracy