import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from dash import Dash, html, dash_table
import joblib


def make_histogram_graphic(data_base_path):
    scores = pd.read_csv(data_base_path)

    avg = []
    for models in scores.columns:
        avg.append([models, scores[models].mean()])

    avg = pd.DataFrame(avg, columns=['model', 'accuracy'])
    figure = px.histogram(avg, title="model's accuracy", x='accuracy', y='model', color='model')
    figure.show()
    print(avg)


def make_table(table_path):
    df = pd.read_csv(table_path)
    external_stylesheets = ['assets/css/style.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div([
        html.Div(children="Model test", style={'fontSize': '24px', 'textAlign': 'center'}),
        dash_table.DataTable(data=df.to_dict('records'),
                             style_cell={'textAlign': 'center'},
                             page_size=15,
                             style_data={
                                 'backgroundColor': 'rgb(30, 30, 30)',
                                 'fontSize': '18px',
                                 'width': '50px'

                             },
                             style_header={
                                 'backgroundColor': 'rgb(50, 50, 50)',
                                 'fontWeight': 'bold',
                                 'border': '1px solid white'
                             })
    ])

    app.run(debug=True)


def export_comparing_table(train_table, model_path, number_of_rows, table_path):
    model = joblib.load(model_path)
    table = pd.read_csv(train_table)

    x, y = separate_table(table, number_of_rows)
    prediction = model.predict(x)

    export_table = {
        'Amount': x['Amount'],
        'Prediction': prediction,
        'Real values': y
    }

    export_table = pd.DataFrame(export_table)
    export_table.to_csv(table_path, index=False)


def export_train_table(accuracies, model_names, filename):
    while True:
        try:
            old_score = pd.read_csv(filename)
            break
        except pd.errors.EmptyDataError:
            print('error 1')
            old_score = dict()
            for model_name in model_names:
                old_score.update({model_name: []})
            old_score = pd.DataFrame(old_score)
            break
        except FileNotFoundError:
            print('error 2')
            old_score = open(filename, 'a')
            old_score.close()

    new_score = pd.DataFrame()

    for accuracy, modelo in zip(accuracies, model_names):
        new_score[modelo] = [accuracy]

    score = pd.concat([old_score, new_score])
    score.to_csv(filename, index=False)


def separate_table(table, number_of_rows):
    table = table.sample(n=number_of_rows)
    x = table.drop(columns='Class')
    y = table['Class']

    return x, y


def train_one_model(train_table, model, number_of_rows):
    x, y = separate_table(train_table, number_of_rows)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model.fit(x_train, y_train)

    if model == KNeighborsClassifier():
        prediction = model.predict(x_test.to_numpy())
    else:
        prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, prediction) * 100.0

    return {'model': model, 'accuracy': accuracy}


def train_all_models(data_base_path, accuracy_path, number_of_rows, number_of_trains=1):
    models = (RandomForestClassifier(),
              GradientBoostingClassifier(),
              KNeighborsClassifier(),
              RidgeClassifier(),
              Perceptron(),
              DecisionTreeClassifier())

    model_names = ('RandomForestClassifier',
                   'GradientBoostingClassifier',
                   'KNeighborsClassifier',
                   'RidgeClassifier',
                   'Perceptron',
                   'DecisionTreeClassifier')

    table = pd.read_csv(data_base_path)
    table = table.drop(columns='id')

    for i in range(number_of_trains):
        print(f'Starting train #{i + 1}\n')
        accuracies = []

        for model, model_name in zip(models, model_names):
            print(f'#{i + 1} - Training {model_name}')

            accuracy = train_one_model(table, model, number_of_rows)['accuracy']
            accuracies.append(accuracy)

            print(f'#{i + 1} - Score: {accuracy}/100')
            print()
        print('-' * 12, end='\n\n')

        export_train_table(accuracies, model_names, accuracy_path)


def main():
    # make_histogram_graphic('score-100-trains-40000-db-rows.csv')
    # export_comparing_table('creditcard_2023.csv',
    #                        'trained_model.joblib',
    #                        15,
    #                        'comparing-model-results.csv')
    make_table('comparing-model-results.csv')


if __name__ == '__main__':
    main()
