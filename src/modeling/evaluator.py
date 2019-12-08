from sklearn.metrics import f1_score  
from imblearn.metrics import geometric_mean_score

def evaluate(true_data, predicted_data):
    result = dict()
    #TODO: consider the case of multi labels.
    #notice that both functions take the same parameters 

    #TODO: might need to use lables parameter
    for av in [None ,'micro', 'macro', 'weighted']:
        f1 = f1_score(true_data,predicted_data, average=av)
        g_mean = geometric_mean_score(true_data, predicted_data, average=av)
        result.update({"f_measure_" + str(av) : f1, "g_mean_" + str(av): g_mean})
    return result

y_true = [0, 1, 0, 0, 1, 1, 2]
y_pred = [0, 0, 1, 0, 0, 1, 3]
print(evaluate(y_true, y_pred))

