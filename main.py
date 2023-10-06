from functions import (train_all_models,
                       train_one_model,
                       make_histogram_graphic,
                       make_table)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
from IPython.display import display
import joblib


def main():
    output_file = 'score-test.csv'
    # train_all_models('creditcard_2023.csv',
    #                  output_file,
    #                  30_000,
    #                  5)

    # make_histogram_graphic(output_file)

    output_file = pd.read_csv('creditcard_2023.csv')

    print('Training last model')
    random_forest_model = train_one_model(output_file, RandomForestClassifier(), 40_000)['model']
    perceptron_model = train_one_model(output_file, Perceptron(), 40_000)['model']

    print('Exporting last model')
    joblib.dump(random_forest_model, 'RandomForest-40000.joblib')
    joblib.dump(perceptron_model, 'Perceptron-40000.joblib')
    print('The model RandomForestClassifier was exported successfully.')
    print('The model Perceptron was exported successfully.')
    display(pd.read_csv("comparing-model-results.csv"))

make_table('comparing-model-results.csv')

if __name__ == '__main__':
    main()
    make_table()
