import numpy as np
import matplotlib.pyplot as plt

class Polynom(object):
    def __init__(self,coef,degree,constant):
        self.coeff = coef
        self.degree = degree
        self.constant = constant
    
    def power_list(self):
        step = 1
        start = 1
        return np.flip(np.arange(0,self.degree)*step+start)

    def get_values(self,min=-50,max=50,number =1000):
        x = np.linspace(min,max,number)
        y_list = np.zeros(len(x))
        for i in range(len(x)):
            y_list[i] = np.sum([x[i]**p for p in self.power_list()])+self.constant
        return x,y_list
    


    def curve_show(self):
        x,y = self.get_values()
        plt.figure()
        plt.plot(x,y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    p =Polynom([1,0],2,0)
    print(p.power_list())
    p.curve_show()