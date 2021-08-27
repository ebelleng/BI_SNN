# SNN: Training via BackPropagation

import pandas     as pd
import numpy      as np
import my_utility as ut
	
#Training of SNN
def train_snn(x,y,param):
    w1,w2 = ut.iniW(param[0], x, y)  
    cost  = []
    for iter in range(param[1]):
        #Step 1: Forward
        Act = ut.forward(x,w1,w2) 
      
        #Step2: Backward
        #w1, w2, mse(iter) = snn_backward(Act, y, w1,w2,mu); taza aprendizaje(?)
        # cost.append(mse) 
    return(w1,w2,cost) 
   
# Beginning ...
def main():
    par_snn         = ut.load_config()    
    xe,ye           = ut.load_data('train.csv')        
    w1,w2, cost     = train_snn(xe,ye,par_snn)         
    ut.save_w(w1,w2,'w_snn.npz',cost,'costo.csv')
       
if __name__ == '__main__':   
	 main()

