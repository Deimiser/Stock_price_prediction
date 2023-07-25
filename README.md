# Stock_price_prediction

This code is a Python script that performs various tasks related to analyzing and visualizing stock prices of Tesla and building a simple linear regression model for predicting stock prices. I'll explain each line of code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
These are import statements for necessary libraries. Pandas is used for data manipulation, NumPy for numerical computations, and Matplotlib for data visualization. The `%matplotlib inline` command ensures that the plots will be displayed directly in the notebook.

```python
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
```
These imports are for using Plotly for interactive data visualization. Plotly is a library that provides more advanced and interactive plots compared to Matplotlib.

```python
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
```
This sets up Plotly to work in the notebook and allows for offline plotting.

```python
tesla = pd.read_csv(r'C:\Users\Saurab\Desktop\STOCK\tesla.csv')
tesla.head()
```
This loads the Tesla stock data from the CSV file located at 'C:\Users\Saurab\Desktop\STOCK\tesla.csv' into a Pandas DataFrame and displays the first few rows using `head()`.

```python
tesla.info()
```
This displays information about the DataFrame, including the data types of columns and non-null counts.

```python
tesla['Date'] = pd.to_datetime(tesla['Date'])
```
This converts the 'Date' column in the DataFrame to a datetime data type, which allows for easier manipulation and plotting of time-series data.

```python
print(f'Dataframe contains stock prices between {tesla.Date.min()} {tesla.Date.max()}') 
print(f'Total days = {(tesla.Date.max()  - tesla.Date.min()).days} days')
```
These lines print the minimum and maximum dates present in the DataFrame and calculate and print the total number of days covered by the dataset.

```python
tesla.describe()
```
This provides a statistical summary of the DataFrame, including count, mean, standard deviation, minimum, and maximum values for each numeric column.

```python
tesla[['Open','High','Low','Close','Adj Close']].plot(kind='box')
```
This line generates a box plot to visualize the distribution of 'Open', 'High', 'Low', 'Close', and 'Adj Close' prices.

```python
layout = go.Layout(
    title='Stock Prices of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
```
This defines a custom layout for the Plotly plot that will be created later. It specifies the title and axis labels with their font properties.

```python
tesla_data = [{'x':tesla['Date'], 'y':tesla['Close']}]
plot = go.Figure(data=tesla_data, layout=layout)
```
This creates a Plotly figure object using the Tesla data. It uses the 'Date' column as the x-axis and the 'Close' column as the y-axis data. The `layout` defined earlier is also provided to customize the plot appearance.

```python
iplot(plot)
```
This line uses Plotly's `iplot` function to display the plot of Tesla's stock closing prices over time.

```python
X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
```
These lines prepare the data for building the linear regression model. The 'index' of the DataFrame is used as the feature 'X', and the 'Close' column is used as the target 'Y'. The data is split into training and testing sets using the `train_test_split` function from scikit-learn.

```python
scaler = StandardScaler().fit(X_train)
```
This creates a `StandardScaler` object and fits it to the training data 'X_train'. The scaler is used for feature scaling to bring all features to the same scale.

```python
lm = LinearRegression()
lm.fit(X_train, Y_train)
```
This creates a linear regression model using scikit-learn's `LinearRegression` class and fits it to the training data 'X_train' and 'Y_train'.

```python
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data, layout=layout)
```
These lines create two Scatter traces for the training data: one for the actual 'Y_train' values as markers and the other for the predicted values using the trained linear model. The traces are then combined into a Plotly figure 'plot2' with the previously defined custom layout.

```python
iplot(plot2)
```
This line uses Plotly's `iplot` function to display the plot comparing the actual and predicted values for the training dataset.

```python
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)
```
This calculates and prints the evaluation metrics for the model. It calculates and prints the R-squared score and mean squared error (MSE) for both the training and testing datasets. The R-squared score indicates the proportion of variance in the target variable that is predictable from the features, and the MSE represents the average squared difference between the actual and predicted values.
