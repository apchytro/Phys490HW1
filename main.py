# Name: Anthony Chytros
# Student ID: 20624286
# Phys 490 HW1 

#!/usr/bin/python

#import packages
import sys
import os
import numpy as np
import json

#read in flags and create output file name based on them
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = (os.path.splitext(file1)[0]) + ".out"

# Open .in file, .json file and create and open .out file
f_in = open (file1,'r')
f_j = open(file2,'r')
f_out = open(file3,'w')

# Load in hyperparamaters from .json file
jdata = json.load(f_j)
alpha = jdata['learning rate']
num_iter = jdata['num iter']

# Read data from .in
f1 = f_in.readlines()

# Close files now that they have been read
f_in.close()
f_j.close()

# Initialize x matrix and y vector
x = []
y = []

# Create X matrix and Y vector
for i in f1:
    h = i.split()
    length = len(h) 
    ta = [1]
    for j in range(length - 1):
        ta.append(float(h[j]))  
    x.append(ta)
    y.append(float(h[-1]))
    
x = np.array(x)
y = np.array(y)


#Analytic w calculation
w_analytic = x.transpose().dot(x)
temp = np.linalg.inv(w_analytic) 
temp2 = temp.dot(x.transpose())
w_analytic = temp2.dot(y)

#SGD 
w_gd = np.ones(x.shape[1])

for n in range(num_iter):
    index = np.random.randint(0,len(x))
    w_gd += alpha * (y[index] - w_gd.dot(x[index])) * x[index]
    
# Write the w_analytic and w_gd vectors to .out file
for entry in w_analytic:
    f_out.write("{0:.4f}".format(entry))
    f_out.write("\n")
    
f_out.write("\n")

for entry in w_gd:
    f_out.write("{0:.4f}".format(entry))
    f_out.write("\n")
    
f_out.close()
