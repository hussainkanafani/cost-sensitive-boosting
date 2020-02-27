from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(true_data, predicted_data):
    f1 = f1_score(true_data, predicted_data, average='binary')
    precision = precision_score(true_data, predicted_data, average='binary')
    recall = recall_score(true_data, predicted_data, average='binary')

    return f1, precision, recall

