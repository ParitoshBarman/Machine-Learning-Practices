import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



diabetes = datasets.load_diabetes()

# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.keys())
# print(diabetes.data)
# print(diabetes.DESCR)



diabetes_X = diabetes.data

# print(diabetes_X)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-20:]


model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)




print(f"Mean squared error is : {mean_squared_error(diabetes_Y_test, diabetes_Y_predicted)}")
print(f"Weights: {model.coef_}")
print(f"Intercept: {model.intercept_}")



# plot.scatter(diabetes_X_test, diabetes_Y_test)

# plot.plot(diabetes_X_test, diabetes_Y_predicted)

# plot.show()



# Mean squared is error : 2561.3204277283867
# Weights: [941.43097333]
# Intercept: 153.39713623331698