# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# Init.weights of the DL 
def iniW(nodes_hidden, x, y):
    size_input, _ = x.shape # 375, 5
    _, size_output = y.shape # 375, 1
    
    # Init. w1 y w2 
    w1 = np.random.random((nodes_hidden, size_input)) #dim -> (20x5)
    w2 = np.random.random((size_output, nodes_hidden)) #dim -> (1x20)
       
    return(w1,w2)


#STEP 1: Feed-forward of DAE
def forward(x,w1,w2):
    # Calcula la activación de los Nodos Ocultos
    z1 = np.dot(w1, x)
    a1 = act_sigmoid(z1)
    # Calcula la activación de los Nodos de Salida
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)

    return a2

# STEP 2: Gradiente via BackPropagation
def backward(Act, y, w1,w2, mu):
    # Calcular el error
    error = Act - y.T
    # Calcular el gradiente oculto y salida    
    dCdW = grad_bp(Act, w1, w2, error)
    # Actualizar los pesos
    w1, w2 = updW(w1, w2, mu, dCdW)
    # Calcular Error cuadratico medio
    mse = np.sum(error[0] ** 2) / len(error[0])
    
    return w1, w2, mse

def grad_bp(Act, w1, w2, e):
    a2 = Act                # 1, 375
    z2 = deriva_sigmoid(a2) # 1, 375
    a1 = np.dot(w2.T, z2)   # 20, 375
    z1 = deriva_sigmoid(a1) # 20, 375
    x = np.dot(w1.T, z1)    # 5, 375

    # Calcular gradiente capa salida
    delta2 = np.multiply(e, deriva_sigmoid(z2)) # Probar con a2
    dCdW2 = np.dot(delta2, a1.T)
    # Calcular gradiente capa oculta
    delta1 = np.multiply( np.dot(w2.T, delta2), deriva_sigmoid(z1) )
    dCdW1 = np.dot( delta1, x.T)

    return dCdW1, dCdW2

# Update SNN's Weight 
def updW(w1, w2, mu=0.1, dCdW=(0,0)):    
    dCdW1, dCdW2 = dCdW
    # Actualizar pesos ocultos
    w1 = w1 - mu * dCdW1   
    # Actualizar pesos salida
    w2 = w2 - mu * dCdW2
    return w1, w2

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# Métrica
def metrica(x,y):

    #Error valor real - valor estimado
    e =  x - y
    n = len(x)
    
    #Calculo de MAE
    absolute = np.absolute(e)
    mae = 1/n * np.sum(absolute)
    
    #Calculo de MSE Y RMSE
    diffSquare = e**2
    suma = np.sum(diffSquare)
    mse = 1/n * suma
    rmse = np.sqrt(mse)
    
    #Calculo de R2
    varE = e.var()#np.var(e)
    varY = x.var()#np.var(x)
    r2 = 1 - (varE/varY)

    
    #print('mae',mae)
    #print('mse',mse)
    #print('rmse',rmse)
    #print('mse',mse) 
    #print('var e',varE)
    #print('vat y',varY)
    #print('varE/varY',varE/varY)
    #print('R2',r2)
    
    #Guardado en archivo metrica.csv
    archivo = open('metricas.csv', 'w')
    archivo.write('MAE: '+ str(mae))
    archivo.write("\n")
    archivo.write('RMSE: '+ str(rmse))
    archivo.write("\n")
    archivo.write('R2: '+ str(r2))
    archivo.close()
    
    #falta crear el archivo estima.csv
  
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
    # Guardar pesos 
    np.savez_compressed(fname_w, w1, w2)
    # Guardar mse
    archivo = open(fname_cost, 'w')
    [ archivo.write(f'{c}\n') for c in cost ]
    archivo.close()
    
#load weight of SNN in numpy format
def load_w(fname):
    #completar code
    w1 = np.load(fname)['arr_0']
    w2 = np.load(fname)['arr_1']

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
    x = x.T
    y = df_norm.drop(labels=[i for i in range(5)],axis = 1)
    
    return (x.to_numpy(),y.to_numpy())


