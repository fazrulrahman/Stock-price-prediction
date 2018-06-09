import csv 
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
months = []
price = []

def fetch_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[2]))
			months.append(int(row[0].split('-')[1]))
			price.append(float(row[4]))
		# print(dates)
		# print(months)
		# print(price)

	return

def predict_price(dates, months, price, x):
	dates = np.reshape(dates,(len(dates),1))
	months = np.reshape(months,(len(months),1))
	# print(dates)

	svr_linear = SVR(kernel= 'linear', C=1e3)
	svr_linear.fit(dates,price)

	svr_poly = SVR(kernel= 'poly', C=1e3, degree= 2)
	svr_poly.fit(dates,price)

	svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma= 0.1)
	svr_rbf.fit(dates,price)

	plt.scatter(dates, price, color='black', label='data')
	plt.plot(dates, svr_linear.predict(dates), color='green', label='Linear-model')
	plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial-model')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='Polynomial-model')
	plt.legend()
	plt.show()

	return svr_linear.predict(x)[0],svr_poly.predict(x)[0],svr_rbf.predict(x)[0]

fetch_data('AAPL.csv')

predicted_price = predict_price(dates, months, price, 30)
print(predicted_price)