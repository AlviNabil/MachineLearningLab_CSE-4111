import numpy as np

def gradDes(x: np.ndarray, y:np.ndarray)->np.ndarray:

    m_curr = c_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr*x+c_curr
        # print(type(y_predicted))
        cost = (1/(2*n))*sum(val**2 for val in (y_predicted-y))
        m_d = (2/n)*sum(x*(y_predicted-y))
        c_d  = (2/n)*sum(y_predicted-y)
        m_curr = m_curr - learning_rate*m_d
        c_curr = c_curr - learning_rate*c_d
        print("m==> {}, b==> {}, cost==> {}, iteration==> {}".format(m_curr, c_curr, cost,i))



x = np.array([[1,2,3,4,5],[2,3,4,5,6]])
y = [7,12,17,22,27]
gradDes(x,y)

