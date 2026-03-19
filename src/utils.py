import numpy as np
import pandas as pd

def load_query_data(csv_file, column_sequences, column_headers, column_labels=None):
    try:
        df = pd.read_csv(csv_file)
    except ValueError:
        raise ValueError("The query dataset must be a .csv file with {} and {} columns.".format(column_sequences, column_headers))
    assert column_sequences in df.columns and column_headers in df.columns, "The input .csv file must contain {} and {} columns.".format(column_sequences, column_headers)
    seq_query = df[column_sequences].values
    headers_query = df[column_headers].values
    labels_query = df[column_labels].values if column_labels in df.columns else None
    return seq_query, headers_query, labels_query