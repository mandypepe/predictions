import requests
import airflow
import csv

from sklearn.linear_model import LinearRegression
from pmdarima import arima as am
from statsmodels.tsa.arima import model as ari
from airflow.models import DAG
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np
from airflow.utils.trigger_rule import TriggerRule
from datetime import timedelta
import pendulum

Moneda = "_btc"
minvals=10
config = {
    "cod": "PREDICTIONS",
    "empresa": "CUBA",
    "area": "BTC",
    "nombre": "BTC",
}

config["dag_id"] = config['cod'] + '-' + config['nombre']

args = {
    'owner': 'Armando',
    'start_date': airflow.utils.dates.days_ago(1),
}

dag = DAG(
    dag_id=config["area"] + '-' + config["cod"],
    catchup=False,
    default_args=args,
    max_active_tasks=3,
    max_active_runs=1,
    schedule_interval='* * * * *',
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    dagrun_timeout=timedelta(minutes=120),
)
def validateplustow(series):
    if len(series) < minvals:
        raise ValueError(f"The series must have at least {minvals} values")

def read_csv_file(name):
    with open(f"dags/files/prediciones/{name}.csv", newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        # Create an empty list to store the data
        pvalues = []
        # Loop through each row in the CSV file and append it to the list
        for row in reader:
            pvalues.append(float(row[0]))
    return pvalues

def save_csv_file(name,values):
    with open(f"dags/files/prediciones/{name}.csv", 'a+') as wr:
        newline = f"{values}"
        wr.write(newline)
        wr.write("\n")
        wr.close()

def _KNeighborsRegressor(series, window_size=2, n_neighbors=2):
    validateplustow(series)
    series = np.array(series)
    # Split series into features and target
    X = []
    y = []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)

    # Fit K-NN model to series
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict([series[-window_size:]])[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")

    return next_value


def _RandomForestRegressor(series):
    validateplustow(series)
    # Split series into features and target
    series = np.array(series)
    # Split series into features and target
    X = np.arange(len(series)).reshape(-1, 1)
    y = series

    # Fit random forest model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict([[len(series)]])[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def _GradientBoostingRegressor(series):
    validateplustow(series)
        # Split series into features and target
    series = np.array(series)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series

    # Fit gradient boosting model
    model = GradientBoostingRegressor()
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict([[len(series)]])[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def _svms(series):
    validateplustow(series)
    # Split series into features and target
    series = np.array(series)
    # Split series into features and target
    X = np.arange(len(series)).reshape(-1, 1)
    y = series

    # Fit SVM model
    model = SVR()
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict([[len(series)]])[0]

    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def _DecisionTreeRegressor(series):
    validateplustow(series)
    # Split series into features and target

    series = np.array(series)
    # Split series into features and target
    X = np.arange(len(series)).reshape(-1, 1)
    y = series

    # Fit decision tree model
    model = DecisionTreeRegressor()
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict([[len(series)]])[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def _Poisson(series):
    validateplustow(series)
    # Split series into features and target
    series=list(map(round,series))
    numpy_array = np.array(series)
    X = np.arange(1, len(numpy_array) + 1)
    X = sm.add_constant(X)
    # Fit Poisson regression model
    model = sm.GLM(numpy_array, X, family=sm.families.Poisson())
    result = model.fit()
    # Predict next value in series
    next_X = np.array([1, len(numpy_array) + 1])
    next_value = result.predict(next_X)
    print(f"El posible proximo precio de BTC es {next_value[0]} USD")
    return next_value[0]


def _Multiple_Regression(series):
    validateplustow(series)
    # Split series into features and target
    numpy_array = np.array(series)

    X = numpy_array[:-1].reshape(-1, 1)
    y = numpy_array[1:]
    forp=numpy_array[-1].reshape(1, -1)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next value in series
    next_value = model.predict(forp)
    print(f"El posible proximo precio de BTC es {next_value[0]} USD")
    return next_value[0]


def _state_space(series):
    validateplustow(series)
    """
    Predicts the next value in a time series given a sequence of past values using state space models.

    Parameters:
    series (list or numpy.ndarray): A list or numpy array of numbers representing the time series.

    Returns:
    float: The predicted next value in the time series.
    """

    # Convert the input series to a numpy array
    series = np.array(series)

    # Create a state space model from the time series
    model = sm.tsa.UnobservedComponents(series, 'local level')

    # Fit the state space model to the time series
    fitted_model = model.fit()

    # Get the predicted next value in the time series
    next_value = fitted_model.forecast(steps=1)
    print(f"El posible proximo precio de BTC es {next_value[0]} USD")
    return next_value[0]


def _seasonal_decompose(series, freq):
    validateplustow(series)
    """
    Predicts the next value in a time series given a sequence of past values using Seasonal Decomposition.

    Parameters:
    series (list): A list of numbers representing the time series.
    freq (int): The frequency of the time series.

    Returns:
    float: The predicted next value in the time series.
    """

    # Perform seasonal decomposition on the series
    decomposition = seasonal_decompose(series, period=freq)

    # Extract the trend and seasonal components
    trend = decomposition.trend
    seasonal = decomposition.seasonal

    # Subtract the trend and seasonal components from the original series to obtain the residual component
    residual = series - trend - seasonal

    # Use the residual component to predict the next value in the time series
    next_value = trend[-1] + seasonal[-1] + residual[-1]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value


def _holt_winters(series, seasonal_periods=None, trend=None, seasonal=None):
    validateplustow(series)
    """
    Predicts the next value in a series based on the previous values in the same array, using
    the Holt-Winters method.

    Parameters:
    series (list): A list of numbers representing the series.
    seasonal_periods (int): The number of seasonal periods in the data. Default is None.
    trend (str): The type of trend to use in the model. Can be 'add', 'mul', or None. Default is None.
    seasonal (str): The type of seasonal component to use in the model. Can be 'add', 'mul', or None. Default is None.

    Returns:
    float: The predicted next value in the series.
    """

    # Check that the series has at least two values


    # Fit the Holt-Winters model to the series
    if seasonal_periods is not None:
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    else:
        model = Holt(series, trend=trend)

    model_fit = model.fit()

    # Predict the next value in the series
    next_value = model_fit.forecast()[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value


def _exponential_smoothing(series):
    """
    Predicts the next value in a series based on the previous values in the same array, using
    ExponentialSmoothing to automatically determine the optimal Holt-Winters smoothing parameters.

    Parameters:
    series (list): A list of numbers representing the series.

    Returns:
    float: The predicted next value in the series.
    """

    # Check that the series has at least two values
    validateplustow(series)

    # Determine the optimal Holt-Winters smoothing parameters using ExponentialSmoothing
    model = ExponentialSmoothing(series)
    model_fit = model.fit()

    # Predict the next value in the series based on the Holt-Winters model
    next_value = model_fit.forecast()[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value


def _auto_arima(series):
    """
    Predicts the next value in a series based on the previous values in the same array, using
    auto_arima to automatically determine the optimal ARIMA order.

    Parameters:
    series (list): A list of numbers representing the series.

    Returns:
    float: The predicted next value in the series.
    """

    # Check that the series has at least two values
    validateplustow(series)

    # Determine the optimal ARIMA order using auto_arima
    model = am.auto_arima(series, suppress_warnings=True)

    # Fit the model to the series
    model.fit(series)

    # Predict the next value in the series based on the ARIMA model
    next_value = model.predict(n_periods=1)[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def arima(series, order):
    """
    Predicts the next value in a series based on the previous values in the same array, using ARIMA.

    Parameters:
    series (list): A list of numbers representing the series.
    order (tuple): A tuple (p, d, q) representing the ARIMA order.

    Returns:
    float: The predicted next value in the series.
    """

    # Check that the series has at least two values
    validateplustow(series)

    # Create an ARIMA model and fit it to the series
    model = ari.ARIMA(series, order=order)
    model_fit = model.fit()

    # Predict the next value in the series based on the ARIMA model
    next_value = model_fit.forecast()[0]
    print(f"El posible proximo precio de BTC es {next_value} USD")
    return next_value

def _load_values(ti):
    """
    Predicts the next value in a series based on the previous values in the same array, using linear regression.

    Parameters:
    series (list): A list of numbers representing the series.

    Returns:
    float: The predicted next value in the series.
    """
    name=Moneda
    pvalues=read_csv_file(name)

    # Print the contents of the list
    ti.xcom_push(key='send_price', value=pvalues)

def _LinearRegression(pvalues):
    # Check that the series has at least two values
    validateplustow(pvalues)

    X = [[i] for i in range(len(pvalues))]
    y = pvalues

    # Create a linear regression model and fit it to the series
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next value in the series based on the linear regression model
    next_value = model.predict([[len(pvalues)]])
    return next_value[0]

def _LinearRegression_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction=_LinearRegression(pvalues)
    ti.xcom_push(key='prediction', value=prediction)
def _ExponentialSmoothing_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction=_exponential_smoothing(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _Arima_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    order=(2,1,2)
    prediction=arima(pvalues,order)
    ti.xcom_push(key='prediction', value=prediction)

def _Auto_Arima_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _auto_arima(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _Holt_winters_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    seasonal_periods= len(pvalues)
    prediction =_holt_winters(pvalues,seasonal_periods,trend=None,seasonal=None)
    ti.xcom_push(key='prediction', value=prediction)

def _seasonal_decompose_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    freq= 1
    prediction =_seasonal_decompose(pvalues, freq)
    ti.xcom_push(key='prediction', value=prediction)

def _state_spac_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction =_state_space(pvalues)
    ti.xcom_push(key='prediction', value=prediction)


def _multiple_regression_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction =_Multiple_Regression(pvalues)
    ti.xcom_push(key='prediction', value=prediction)


def _poisson_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _Poisson(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _DecisionTreeRegressor_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _DecisionTreeRegressor(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _RandomForestRegressor_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _RandomForestRegressor(pvalues)
    ti.xcom_push(key='prediction', value=prediction)


def _svms_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _svms(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _GradientBoostingRegressor_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction = _GradientBoostingRegressor(pvalues)
    ti.xcom_push(key='prediction', value=prediction)

def _KNeighborsRegressor_predict_value(ti):
    pvalues = ti.xcom_pull(task_ids="_load_values", key='send_price')
    prediction =_KNeighborsRegressor(pvalues, window_size=5, n_neighbors=5)
    ti.xcom_push(key='prediction', value=prediction)

def _save_predict_value(ti):
    linearr_regression = ti.xcom_pull(task_ids="_LinearRegression", key='prediction')
    exponential_smoothing = ti.xcom_pull(task_ids="_ExponentialSmoothing_predict_value", key='prediction')
    arima = ti.xcom_pull(task_ids="_Arima_predict_value", key='prediction')
    auto_arima = ti.xcom_pull(task_ids="_Auto_Arima_predict_value", key='prediction')
    holt_winters = ti.xcom_pull(task_ids="_Holt_winters_predict_value", key='prediction')
    seasonal_decompose = ti.xcom_pull(task_ids="_seasonal_decompose_predict_value", key='prediction')
    state_spac = ti.xcom_pull(task_ids="_state_spac_predict_value", key='prediction')
    multiple_regression = ti.xcom_pull(task_ids="_multiple_regression_predict_value", key='prediction')
    decision_treeregressor = ti.xcom_pull(task_ids="_DecisionTreeRegressor", key='prediction')
    randomforest_regressor = ti.xcom_pull(task_ids="_RandomForestRegressor", key='prediction')
    svms = ti.xcom_pull(task_ids="_svms_predict_value", key='prediction')
    gradientboosting_regressor = ti.xcom_pull(task_ids="_GradientBoostingRegressor_predict_value", key='prediction')
    kneighbors_regressor = ti.xcom_pull(task_ids="_KNeighborsRegressor", key='prediction')
    save_csv_file(name="linearr_regression",values=linearr_regression)
    save_csv_file(name="exponential_smoothing",values=exponential_smoothing)
    save_csv_file(name="arima",values=arima)
    save_csv_file(name="auto_arima",values=auto_arima)
    save_csv_file(name="holt_winters",values=holt_winters)
    save_csv_file(name="seasonal_decompose",values=seasonal_decompose)
    save_csv_file(name="state_spac",values=state_spac)
    save_csv_file(name="multiple_regression",values=multiple_regression)
    save_csv_file(name="decision_treeregressor",values=decision_treeregressor)
    save_csv_file(name="randomforest_regressor",values=randomforest_regressor)
    save_csv_file(name="svms",values=svms)
    save_csv_file(name="gradientboosting_regressor",values=gradientboosting_regressor)
    save_csv_file(name="kneighbors_regressor",values=kneighbors_regressor)
    print(f"Mi prediccion del siguiente valor es:{linearr_regression} dijo: Linear Regression ")
    print(f"Mi prediccion del siguiente valor es:{exponential_smoothing} dijo:Exponential Smoothing ")
    print(f"Mi prediccion del siguiente valor es:{arima} dijo: ARIMA")
    print(f"Mi prediccion del siguiente valor es:{auto_arima} dijo: AUTO ARIMA ")
    print(f"Mi prediccion del siguiente valor es:{holt_winters} dijo: HOLT WINTERS")
    print(f"Mi prediccion del siguiente valor es:{seasonal_decompose} dijo: Seasonal Decompose")
    print(f"Mi prediccion del siguiente valor es:{state_spac} dijo: State Space ")
    print(f"Mi prediccion del siguiente valor es:{multiple_regression} dijo: Multiple Regression ")
    print(f"Mi prediccion del siguiente valor es:{decision_treeregressor} dijo: Decision Tree Regressor")
    print(f"Mi prediccion del siguiente valor es:{randomforest_regressor} dijo: Random Forest Regressor")
    print(f"Mi prediccion del siguiente valor es:{svms} dijo: Epsilon-Support Vector Regression ")
    print(f"Mi prediccion del siguiente valor es:{gradientboosting_regressor} dijo: Gradient Boosting Regressor")
    print(f"Mi prediccion del siguiente valor es:{kneighbors_regressor} dijo: KNeighbors Regressor")




def get_price():
        url = 'https://api.coindesk.com/v1/bpi/currentprice.json'
        response = requests.get(url)
        data = response.json()
        precio_btc=data['bpi']['USD']['rate_float']
        print(f"El precio actual de BTC es {precio_btc} USD")
        name_file = Moneda
        values=precio_btc
        save_csv_file(name_file,values)





t_get_currencies = PythonOperator(
    task_id='_get_currencies',
    python_callable=get_price,
    provide_context=True,
    dag=dag)

t_load_values = PythonOperator(
    task_id='_load_values',
    python_callable=_load_values,
    provide_context=True,
    dag=dag)

t_LinearRegression_predict_value = PythonOperator(
    task_id='_LinearRegression',
    python_callable=_LinearRegression_predict_value,
    provide_context=True,
    dag=dag)

t_ExponentialSmoothing_predict_value = PythonOperator(
    task_id='_ExponentialSmoothing_predict_value',
    python_callable=_ExponentialSmoothing_predict_value,
    provide_context=True,
    dag=dag)

t_Arima_predict_value = PythonOperator(
    task_id='_Arima_predict_value',
    python_callable=_Arima_predict_value,
    provide_context=True,
    dag=dag)
t_Auto_Arima_predict_value = PythonOperator(
    task_id='_Auto_Arima_predict_value',
    python_callable=_Auto_Arima_predict_value,
    provide_context=True,
    dag=dag)
t_Holt_winters_predict_valu = PythonOperator(
    task_id='_Holt_winters_predict_value',
    python_callable=  _Holt_winters_predict_value,
    provide_context=True,
    dag=dag)

t_seasonal_decompose_predict_value = PythonOperator(
    task_id='_seasonal_decompose_predict_value',
    python_callable=_seasonal_decompose_predict_value,
    provide_context=True,
    dag=dag)

t_state_spac_predict_value = PythonOperator(
    task_id='_state_spac_predict_value',
    python_callable=_state_spac_predict_value,
    provide_context=True,
    dag=dag)

t_multiple_regression_predict_value= PythonOperator(
    task_id='_multiple_regression_predict_value',
    python_callable=_multiple_regression_predict_value,
    provide_context=True,
    dag=dag)

#t_poisson_predict_value= PythonOperator(
#    task_id='_poisson_predict_value',
#    python_callable=_poisson_predict_value,
#    provide_context=True,
#    dag=dag)

t_DecisionTreeRegressor= PythonOperator(
    task_id='_DecisionTreeRegressor',
    python_callable=_DecisionTreeRegressor_predict_value,
    provide_context=True,
    dag=dag)

t_RandomForestRegressor= PythonOperator(
    task_id='_RandomForestRegressor',
    python_callable=_RandomForestRegressor_predict_value,
    provide_context=True,
    dag=dag)

t_svms_predict_value= PythonOperator(
    task_id='_svms_predict_value',
    python_callable=_svms_predict_value,
    provide_context=True,
    dag=dag)

t_GradientBoostingRegressor_predict_value= PythonOperator(
    task_id='_GradientBoostingRegressor_predict_value',
    python_callable=_GradientBoostingRegressor_predict_value,
    provide_context=True,
    dag=dag)

t_KNeighborsRegressor_predict_value= PythonOperator(
    task_id='_KNeighborsRegressor',
    python_callable=_KNeighborsRegressor_predict_value,
    provide_context=True,
    dag=dag)

t_save_predict_value= PythonOperator(
    task_id='_save_predict_value',
    python_callable=_save_predict_value,
    provide_context=True,
    dag=dag)



chain(t_get_currencies,t_load_values,[t_LinearRegression_predict_value,t_ExponentialSmoothing_predict_value,
                                  t_Arima_predict_value,t_Auto_Arima_predict_value,t_Holt_winters_predict_valu,
                                  t_seasonal_decompose_predict_value,t_state_spac_predict_value,
                                  t_multiple_regression_predict_value,t_DecisionTreeRegressor,
                                  t_RandomForestRegressor,t_svms_predict_value,t_KNeighborsRegressor_predict_value,
                                  t_GradientBoostingRegressor_predict_value],t_save_predict_value)
