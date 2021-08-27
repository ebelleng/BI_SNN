# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Init.weights of the DL 
def iniW(nodes_hidden, x, y):
    _, len_x = x.shape # 375, 5
    len_y, _ = y.shape # 1, 375
    # Init. w1 y w2 
    w1 = np.random.random((nodes_hidden, len_x)) #dim -> (20x5)
    w2 = np.random.random((len_y, nodes_hidden)) #dim -> (1x20)
       
    return(w1,w2)


#STEP 1: Feed-forward of DAE
def forward(x,w1,w2):
    # Calcula la activación de los Nodos Ocultos
    z1 = np.dot(w1, x.T)
    a1 = act_sigmoid(z1)
    # Calcula la activación de los Nodos de Salida
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)

    return a2.T    

# STEP 2: Gradiente via BackPropagation
def grad_bp(...):
    #completar code    
    return(...)    

# Update SNN's Weight 
def updW(...):    
    #completar code    
    return(...)

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# Métrica
def metrica(x,y):
    #completar code
    return(...)
  
#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the DL
def load_config():      
    param = np.genfromtxt("config.csv",delimiter=',',dtype=None)    
    par=[]    
    par.append(np.int16(param[0])) # Number of nodes
    par.append(np.int16(param[1])) # Max. Iterations
    par.append(np.float(param[2])) # Learn rate    
    return (par)
# Load data 
def load_data(fname):
    df = pd.read_csv(fname, header=None)   
    df = df.sample(frac=1).reset_index(drop=True) # Reordenar valores aleatoriamente.

    x, y = norm(df) # Normalizar

    return (x, y)

#save weights of SNN in numpy format
def save_w(w1,w2, fname_w, cost, fname_cost):    
    #completar code    
    return
    
#load weight of SNN in numpy format
def load_w(fname):
    #completar code
    return (w1,w2)      

# function normalize
def norm(df):
    a = 0.01
    b = 0.99
    df_norm = pd.DataFrame()

    for col in df:
        min_ = df[col].min()
        max_ = df[col].max()

        df_norm[col] = (df[col] - min_) * (b - a) / (max_ - min_) + a
    
    x = df_norm.drop(labels=[5], axis=1)
    y = df_norm.drop(labels=[i for i in range(5)],axis = 1)

    return (x.to_numpy(),y.to_numpy())

