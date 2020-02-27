class CategoricalImputer:
    def __init__(self, missing_values, strategy='most_frequent'):
        self.missing_values = missing_values
        self.strategy = strategy

    def fit_transform(self, df):
        """ impute a missing value with the most frequent value in the same column """
        for column in df.columns:
            frequent = df[column].mode(dropna=True)[0]
            df[column].replace(to_replace=self.missing_values, value=frequent, inplace=True)
        return df