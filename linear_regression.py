import csv
import matplotlib.pyplot as plt

def prediction(x, w0, w1):
	return w0*x + w1

def calculate_mean(x0):
 	return sum(x0)/len(x0)

if __name__ == '__main__':
    y0 = []
    x0 = []
    numerator = 0
    denominator = 0 
    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            x_read = float(row[0])
            y_read = float(row[1])
            x0.append(x_read)
            y0.append(y_read)
        #print(x0)
        #print(y0)
    mean_x = calculate_mean(x0)
    mean_y = calculate_mean(y0)
    for x,y in zip(x0,y0):
    	x_deviation = x - mean_x
    	y_deviation = y - mean_y

    	numerator += y_deviation*x_deviation
    	denominator += x_deviation**2

    	w0 = numerator/denominator
    	w1 = mean_y - w0 * mean_x

    print(prediction(1714,w0,w1))
    print(prediction(1972,w0,w1))
    print(prediction(1620,w0,w1))
    print(prediction(1000,w0,w1))

    plt.scatter(x0, y0, color='blue', label='Data Points')
    x_range = [min(x0), max(x0)]
    y_range = [prediction(x, w0, w1) for x in x_range]
    plt.plot(x_range, y_range, color='red', label='Regression Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression for 2 dimensional data')
    plt.legend()
    plt.show()
    