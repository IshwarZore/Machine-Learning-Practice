import pandas as pd
import numpy as np

# Expected answer. m = 0.05168176, b=18.0465

def gradient_descent(x, y, lr=0.01, epochs=7000):
    b=0.0
    m=0.0
    
    for epoch in range(epochs):
        yp=m*x+b
        error=y-yp
        cost=np.mean(error**2)
        
        dm=-2*np.mean(error*x)
        db=-2*np.mean(error)
        
        b=b-db*lr
        m=m-dm*lr
        print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")
    return b,m
 



if __name__ == "__main__":
    
    x = np.array([1,2,3,4,5])
    y = np.array([10,15,20,25,30])

    b, m = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")

