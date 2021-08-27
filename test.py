# SNN: Testing 
import pandas as pd
import numpy as np
import my_utility as ut

def main():		
	xv, yv = ut.load_data('test.csv')			
	w1,w2  = ut.load_w('w_snn.npz')
	zv     = ut.forward(xv,w1,w2)      		
	ut.metrica(yv,zv) 	

if __name__ == '__main__':   
	 main()

