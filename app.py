import flask
from flask import render_template, request, jsonify
import pickle
import plotly.express as px
from plotly.io import to_html
from datetime import datetime, timedelta


# MODELO MULTIPLE

def load_model1():
    with open(r'models\finished_model_arima_multiple.model', "rb") as archivo_entrada:
        list_models = pickle.load(archivo_entrada)
        # print(list_models)
    return list_models


list_models = load_model1()


def make_predictions1(list_models, n_periods=2):
    prediccion_model1 = []

    for i in list_models:
        prediccion = i.predict(n_periods=n_periods)
        prediccion_model1.append(list(prediccion))

    # print(prediccion_model1)
    return prediccion_model1


# MODELO UNICO

def load_model2():
    with open(r'models\finished_model_arima.model', "rb") as archivo_entrada:
        modelo_arima = pickle.load(archivo_entrada)
        # print(modelo_arima)
    return modelo_arima


def make_predictions2(modelo_arima, n_periods=2):
    prediccion_model2 = list(modelo_arima.predict(n_periods=n_periods))
    # print(prediccion_model2)
    return prediccion_model2


def get_dates(periods):
    last_date = datetime.strptime('2020-12-31 23:00', '%Y-%m-%d %H:%M')
    min_diff = 240  # 4 hours
    dates = []

    for i in range(1, periods + 1):
        time_change = timedelta(minutes=min_diff * i)
        new_date = last_date + time_change
        dates.append(new_date.strftime('%Y-%m-%d %H:%M'))

    return dates


def make_graph(predictions, periods):
    x = get_dates(periods)
    y = predictions

    fig = px.line(x=x, y=y, title='Particles evolution')

    return to_html(fig, include_plotlyjs=False, include_mathjax=False, full_html=False)


modelo_arima = load_model2()

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/api/model/multiple', methods=['GET'])
def predict1():
    modelo1 = load_model1()
    prediccion_model1 = make_predictions1(modelo1)

    return jsonify(prediccion_model1)


@app.route('/api/model/unico/<int:periods>', methods=['GET'])
def predict2(periods):
    modelo2 = load_model2()
    prediccion_model2 = make_predictions2(modelo2, int(periods))

    response = {
        'prediction': prediccion_model2,
        'graph': make_graph(prediccion_model2, periods)
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
