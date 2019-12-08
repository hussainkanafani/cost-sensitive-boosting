import pandas as pd
import os

def read_data(dataset_path):
        df = pd.read_csv(dataset_path, header=0, sep=';')
        return df

def write_data(df, des_path):
    df.to_csv(des_path, index=False, float_format='%.3f', sep=',')

def put_class_column_first(data):
    columns = data.columns.tolist()
    columns = [columns[-1]] + columns[:len(columns) - 1]
    data = data[columns]
    return data

def change_header_names(data):
    columns = data.columns.tolist()
    names = ['class'] + [x.lower().replace('"', '') for x in columns[1:]] 
    data.columns = names
    return data

def encode_class_labels(data):
    data['class'] = data['class'].map({'yes': 1, 'no': 0})
    return data

def adjust_missing_values(data):
    data = data.replace('unknown', '?')
    return data

def main():
    raw_path = 'D:\Informatik\Master\\1. Semester\Data Mining Labor\\adac123\data\\raw'
    processed_path = 'D:\Informatik\Master\\1. Semester\Data Mining Labor\\adac123\data\processed'
    file = 'bank-additional-full.csv'

    data = read_data(os.path.join(raw_path, file))
    data = put_class_column_first(data)
    data = change_header_names(data)
    data = encode_class_labels(data)
    data = adjust_missing_values(data)
    write_data(data, os.path.join(processed_path, file))

if __name__ == "__main__":
    main()