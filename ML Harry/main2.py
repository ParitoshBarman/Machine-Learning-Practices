import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



x_value = np.array([[1],[2],[3]])

x_train = x_value
x_test = x_value


y_train = np.array([3,2,4])
y_test = np.array([3,2,4])


model = linear_model.LinearRegression()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)



print(f"Mean squred error is --> {mean_squared_error(y_test, y_predict)}")
print(f"Weight --> {model.coef_}")
print(f"Intercept --> {model.intercept_}")


plot.scatter(x_test, y_test)
plot.plot(x_test, y_predict)
plot.show()