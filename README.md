# Cost-sensitive boosting for classification of imbalanced data

## Students
* Omar Arab Oghli
* Hussain Kanafani
* Muaid Mughrabi

## How to Run
```
git clone ....
cd cost-sensitive-boosting
pip install requirements.txt
python src/app.py
```

## Configure the App
The `config.json` file must be stored in the same directory where the `app.py` is. 


```json
{
    "app": {
        "cost_setup": [], // list of costs to loop over
        "ratio_setup": [], // list of imbalance ratios to loop over
        "ratios_cost_setup": [], // list of costs to loop over when app mode is imbalance_ratio
        "n_experiments": 10, // number of experiments to be done in order to compute the average of the computed measures
        "imbalance_ratio": false // if true, only the measures_vs_ratios will be computed and stored. Otherwise, measures_vs_costs and weights_vs_iterations will be computed and stored
    },
    "dataProcessor": {
        "dataDir": "path/to/datasets/dir",
        "testSetSize": 0.2, // 20% of the dataset will be splitted to test set
        "datasets": [ // list of dataset objects
            {
                "filename": "breast-cancer.data", // dataset file name
                // list of columns names that contains categorical feautures
                "categorical_features": ["age","menopause", "tumor-size","inv-nodes","node-caps","deg-malig"
                                        ,"breast","breast-quad","irradiat"]
            },
            // ...
        ]
    },
    "model": {
        "algorithms": ["adac1"], // list of possible algorithms [adaboost, adac1, adac2, adac3, adacost]
        "base_estimator": "DecisionTreeClassifier",
        "n_estimators": 10, // number of iterations that the algorithm has to do
        "learning_rate": 1.0
    }
}
```