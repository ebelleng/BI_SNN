# SNN: Training via BackPropagation

import pandas     as pd
import numpy      as np
import my_utility as ut
	
#Training of SNN
def train_snn(x,y,param):
    w1,w2 = ut.iniW(param[0], x, y)  
    cost = []
    data = (x,y)
    for iter in range(param[1]):
        #Step 1: Forward
        Act = ut.forward(x,w1,w2) 

        #Step2: Backward
        w1, w2, mse = ut.backward(Act, data, w1, w2, param[2]) 
        cost.append(mse)

    return(w1,w2,cost) 
   
# Beginning ...
def main():
    par_snn         = ut.load_config()    
    xe,ye           = ut.load_data('train.csv')        
    w1,w2, cost     = train_snn(xe,ye,par_snn)         
    ut.save_w(w1,w2,'w_snn.npz',cost,'costo.csv')
       
if __name__ == '__main__':   
	 main()

