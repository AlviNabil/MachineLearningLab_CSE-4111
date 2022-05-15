import numpy as np

class linReg:
    def __init__(self) -> None:
        pass

    def gradDes(self,x: np.ndarray, y:np.ndarray)->np.ndarray:
        print(x)
        print(y)
        c_curr = 0
        m_curr = np.zeros(x.shape[0])
        iterations = 1000000
        n = int(x.shape[1])
        learning_rate = 0.013
        parameters = np.zeros(x.shape[0])
        for i in range(iterations):
            y_predicted = 0
            for p in range(len(x)):
                y_predicted += x[p]*m_curr[p]
            y_predicted += c_curr
            
            cost = (1/(2*n))*sum(val**2 for val in (y_predicted-y))
            m_d = np.zeros(len(x))
            for p in range(len(x)):
                m_d[p] = (1/n)*sum(x[p]*(y_predicted-y))
        
            c_d  = (1/n)*sum(y_predicted-y)
            m_curr = m_curr - learning_rate*m_d
            c_curr = c_curr - learning_rate*c_d
            # print(c_curr)
            # print("cost==>","m_d==>", cost, m_curr,c_curr)
        # print(type(parameters), type(m_curr), type(c_curr))
        self.coeff_ = m_curr
        self.intercept_ = c_curr
        parameters = np.concatenate((m_curr, [c_curr]))
            
        return parameters


    # x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]])
    # print(x.shape[0])
    # y = np.array([4,7,10,13,16])
    # print(gradDes(x,y))