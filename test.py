import pandas as pd
import matplotlib.pyplot as plt


# plt.scatter(data.km, data.price)
plt.show()

# def loss_function(m, b, data):
#     total_error = 0
#     for i in range(len(data)):
#         x = data.iloc[i].km
#         y = data.iloc[i].price
#         total_error += (y - (m * x + b)) ** 2
#     return total_error / float(len(data))

def gradient_descent(m_now, b_now, data, L):
    m_grad = 0
    b_grad = 0
    n = len(data)
    
    for i in range(n):
        x = data.iloc[i].km
        y = data.iloc[i].price
        
        m_grad += -(2/n) * x * (y - (m_now * x + b_now))
        b_grad += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_grad * L
    b = b_now - b_grad * L
    return m, b


data = pd.read_csv('data.csv')
m = 0
b = 0
L = 0.001
epochs = 1000

for i in range(40):
    m, b = gradient_descent(m, b, data, L)
print(m, b)


# plt.scatter(data.km, data.price, color='black')
# plt.plot(list(range(100)), [b0 * x + b1 for x in range(0, 100)], color='red')
# # plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color='red')
# plt.show()



