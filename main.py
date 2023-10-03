from functions import (train_all_models,
                       train_one_model,
                       make_histogram_graphic)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib


def main():
    output_file = 'score-test.csv'
    train_all_models('creditcard_2023.csv',
                     output_file,
                     1_000,
                     1)

    make_histogram_graphic(output_file)

    output_file = pd.read_csv('creditcard_2023.csv')

    trained_model = train_one_model(output_file, RandomForestClassifier(), 30_000)['model']
    joblib.dump(trained_model, 'trained_model.joblib')
    print('The model RandomForestClassifier was exported successfully.')


if __name__ == '__main__':
    main()
