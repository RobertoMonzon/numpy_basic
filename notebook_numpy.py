#!/usr/bin/env python
# coding: utf-8

# # "Numpy"

# ### Import library numpy

# In[1]:


import numpy as np


# ### Numpy version

# In[48]:


np.__version__


# ### Numpy config

# In[49]:


np.show_config()


# ### Get help

# Get the information typing of any function

# In[50]:


np.info(np.array)


# # Array vs List

# ### Define the list and the array

# ###### List

# In[2]:


list_1=[1,2,3]
list_1


# ###### Array

# In[4]:


array_1=np.array([4,5,6])
array_1


# ### Print each element

# ###### List

# In[6]:


for e in list_1:
    print (e)


# ###### Array

# In[8]:


for e in array_1:
    print(e)


# ### Append new element

# ###### List

# Original list

# In[10]:


list_1


# New list

# In[12]:


list_1.append(4)
list_1


# It's possible to add a new element just "summing" the original list + a new element between brackets

# In[14]:


list_1 + [5]


# ###### Array

# Isn't possible to use append in an array and using the + will sum the number we are trying to add to each number in the array.

# Original array

# In[15]:


array_1


# In[16]:


array_1 + np.array([4])


# That's called "broadcasting"

# ## Maths

# #### Sum

# ###### List  + List

# Original list

# In[18]:


list_1


# New list

# In[23]:


list_2= list_1 + [5,6,7,8]
list_2


# If we sum two list the new list will be the result of the second list appended to the first list

# Is necessary to import the library operator and use the function add in order to sum two list 

# In[26]:


from operator import add


# In[28]:


list_3=list(map(add,list_1,[5,6,7,8]))
list_3


# In[37]:


[sum(e) for e in zip(list_1,[5,6,7,8])]


# ###### Array + Array

# Original array

# In[20]:


array_1


# New array

# In[22]:


array_2 = array_1 + np.array([7,8,9])
array_2


# Is not possible to sum a new array if the length is not the same

# ### Basic  math operations

# #### List

# Original list:

# In[38]:


list_1


# In order to add a number to each element it is necessary to create an empty list and use a for loop to add the number to each element and append it to the new list and is the same with all the operators

# Sum

# In[39]:


list_sum=[]
for e in list_1:
    list_sum.append(e + 3)
list_sum


# Rest

# In[42]:


list_rest=[]
for e in list_1:
    list_rest.append(e - 1)
list_rest


# Multiply

# In[45]:


list_mult=[]
for e in list_1:
    list_mult.append(e * 2)
list_mult


# Division

# In[46]:


list_div=[]
for e in list_1:
    list_div.append(e / 2)
list_div


# Power

# In[47]:


list_pow=[]
for e in list_1:
    list_pow.append(e ** 2)
list_pow


# Square root

# In order to use the square root function we gotta import the math library

# In[48]:


import math


# In[50]:


list_sqrt=[]
for e in list_1:
    list_sqrt.append(math.sqrt(e))
list_sqrt


# Exponential

# In[51]:


list_exp=[]
for e in list_1:
    list_exp.append(math.exp(e))
list_exp


# ###### NOTE:
# <p>
# There is a lot of math operation we can use but essentialy all are the same

# ###### Array

# Original array

# In[52]:


array_1


# Sum

# In[53]:


array_1 + 2


# Rest

# In[55]:


array_1 - 1


# Multiplication

# In[56]:


array_1 * 3


# Division

# In[57]:


array_1 / 2


# Power

# In[58]:


array_1 ** 2


# alternatively we can use the power function included in the numpy library

# In[61]:


np.power(array_1,2)


# Exponential

# In[62]:


np.exp(array_1)


# ###### NOTE:
# <p>
# There is a lot of functions available in the numpy library

# # The dot product

# ###### Define two arrays

# In[63]:


array_a=np.array([1,2])
array_a


# In[64]:


array_b=np.array([3,4])
array_b


# To use the dot function, declare an empty variable

# In[65]:


dot=0
for a,b in zip(array_a,array_b):
    dot += a*b
dot


# alternatively

# In[67]:


dot=0
for e in range(len(array_a)):
    dot+= array_a[e]*array_b[e]
dot


# Or

# In[68]:


dot=np.sum(array_a*array_b)
dot


# In[69]:


(array_a * array_b).sum()


# Usinig the function dot in the numpy library

# In[71]:


np.dot(array_a,array_b)


# Or

# In[74]:


array_a.dot(array_b)


# In[75]:


array_b.dot(array_a)


# In[76]:


array_a @ array_b


# In[77]:


array_b @ array_a


# # Speed test

# ### Import libraries

# In[78]:


from datetime import datetime


# ###### Define variables

# In[79]:


a=np.random.randn(100)
b=np.random.randn(100)
t=100000


# ###### List

# Define the for loop that we will use to calculate de dot product in a list

# In[80]:


def slow_dot_product(a,b):
    result=0
    for e,f in zip(a,b):
        result += e*f
        return result


# Create a variable to save the current time and then rest the time when the code execute the for loop - the time time it finished

# In[87]:


t0=datetime.now()
for i in range(t):
    slow_dot_product(a,b)
dt1=datetime.now()-t0
dt1.total_seconds()


# ###### Array

# Create a variable to save the current time and then rest the time when the code execute the for loop - the time time it finished

# In[86]:


t0= datetime.now()
for i in range(t):
    a.dot(b)
dt2=datetime.now()-t0
dt2.total_seconds()


# Make a division to calculate the diference in seconds

# In[95]:


dt1.total_seconds()/dt2.total_seconds()


# ###### Run the code at the same time to calculate it

# In[94]:


t0=datetime.now()
for i in range(t):
    slow_dot_product(a,b)
dt1=datetime.now()-t0

t0= datetime.now()
for i in range(t):
    a.dot(b)
dt2=datetime.now()-t0

dt1.total_seconds()/dt2.total_seconds()


# # Matrices

# How would you represent a matrix without numpy? 

# ###### Doing a list of list

# In[96]:


L=[[1,2],[3,4]]
L


# To access to an specific value just type it as a regular list, for example we want the 2 in the first list

# In[97]:


L[0][1]


# ###### Numpy array

# In[99]:


A=np.array([[1,2],[3,4]])
A


# To access to an specific value just type it as a regular list, for example we want the 2 in the first list

# In[100]:


A[0][1]


# Or

# In[102]:


A[0,1]


# Select one column

# In[104]:


A[:,0]


# use A.T to print a 2x2 array

# In[107]:


A.T


# ### Maths

# Can be used in both list and array

# ###### Exponential

# In[108]:


np.exp(A)


# In[109]:


np.exp(L)


# ###### Square Root

# In[110]:


np.sqrt(A)


# In[111]:


np.sqrt(L)


# ###### Multiply arrays

# Create a new array to multiply

# In[114]:


B=np.array([[1,2,3],[4,5,6]])
B


# Array A

# In[115]:


A


# Multiply A*B

# In[117]:


A.dot(B)


# The array has to match the number of columns x num of files

# # Solving linear Systems

# ### Define the arrays

# In[4]:


A=np.array([[1,1],[1.5,4]])
B=np.array([2200,5050])


# In[5]:


A


# In[6]:


B


# ### Solve function

# In[8]:


np.linalg.solve(A,B)


# ### Manually

# In[9]:


np.linalg.inv(A).dot(B)


# # Generating data

# ### Zeros

# In[11]:


np.zeros((2,3))


# ### Ones

# In[13]:


np.ones((5,5))


# ### Identity

# In[18]:


np.identity(6)


# In[20]:


np.eye(6)


# ### Random number

# In[21]:


np.random.random()


# In[23]:


np.random.random((2,3))


# ### Random number mixing negative and positive values

# In[24]:


np.random.randn(2,3)


# ### Random integer numbers

# In[44]:


np.random.randint(60,100,size=(5,5))


# In[42]:


np.random.choice(100,size=(5,5))


# ### Mean,variance and stardard deviation

# In[34]:


R=np.random.rand(10000)


# ###### Mean

# In[35]:


R.mean()


# ###### Mode

# In[39]:


R.var()


# ###### Standard deviation

# In[40]:


R.std()


# # Exercises

# Original link: https://codesolid.com/numpy-practice-questions-to-make-you-an-expert/

# ### 1-D numpy array exercises

# Using a NumPy function, how would you create a one-dimensional NumPy array of the numbers from 10 to 100, counting by 10

# In[114]:


array_10_100_10=np.arange(10,100,10)
array_10_100_10


# How could you create the same NumPy array using a Python range and a list?

# In[118]:


list_10_100_10=[]
for e in range(10,100,10):
    list_10_100_10.append(e)
list_10_100_10


# How might you create a NumPy array of the capital letters, A-Z?

# In[125]:


import string
capital_letters_array=np.array(list(string.ascii_uppercase))
capital_letters_array


# How would you create a ten-element NumPy array object of all zeros

# In[127]:


array_zeros=np.zeros(10)
array_zeros


# What function would return the same number of elements, but of all ones?

# In[128]:


#ones
array_ones=np.ones(10)
array_ones


# How could you create a ten-element array of random integers between 1 and 5 (inclusive)?

# In[129]:


random_array_1_5=np.random.randint(1,5,size=(10))
random_array_1_5


# How can you create a normal distribution of 10 numbers, centered on 5?

# In[131]:


normal_distribution=np.random.normal(5,1,10)
normal_distribution


# What code would create an array of 10 random numbers between zero and one?

# In[136]:


random_number_0_1=np.random.rand(10)
random_number_0_1


# ### Creating and Using Multidimensional Arrays

# Consider an array named “myarray” that is displayed as in the block below. Return the value 7

# * ([[ 1,  2,  3,  4],
# *  [ 5,  6,  7,  8],
# *  [ 9, 10, 11, 12]])

# In[20]:


myarray=(np.arange(1,13)).reshape((3,4))
myarray


# In[22]:


myarray[1][2]


# Given myarray as shown above, what is the dimension of the array?

# In[24]:


myarray.ndim


# An array of three arrays of four elements each like this has twelve elements, of course. How could you create a new array consisting of two arrays of six elements each?

# In[30]:


array_3_4=(np.arange(1,13)).reshape((3,4))
array_3_4


# In[36]:


new_array_3_2=np.array_split(array_3_4,2,axis=1)
arrray_split1=new_array_3_2[0]

arrray_split2=new_array_3_2[1]


# In[37]:


arrray_split1


# In[39]:


arrray_split2


# How could you create a two-dimensional, 3 x 4 array (three arrays of four elements each) with random numbers from 1 to 10?

# In[42]:


random_array_3_4=np.random.randint(1,10,size=(3,4))
random_array_3_4


# How could you create an array of the same size and shape as the previous one, filled with 64-bit integer zeros?

# In[57]:


zeros_array_3_4=(np.zeros(12)).reshape((3,4))
zeros_array_3_4


# Given an array, named "arr“, that looks like:

# * [[0, 1, 2],
# *  [3, 4, 5]]

# How could you display an array that looks like:

# * [[0, 3],
# *  [1, 4],
# *  [2, 5]]

# In[62]:


arr=np.array([[0,1,2],[3,4,5]])
arr


# In[66]:


arr.transpose()


# ### Indexing and Slicing Two-Dimensional Arrays

# Given the following four_by_five array. Write a statement that prints the first row. (It will be a five-element array)

# * [[ 1,  2,  3,  4,  5],
# *  [ 6,  7,  8,  9, 10],
# *  [11, 12, 13, 14, 15],
# *  [16, 17, 18, 19, 20]]

# In[80]:


four_by_five=(np.arange(1,21)).reshape((4,5))
four_by_five


# In[81]:


four_by_five[0]


# Using the same four_by_five array.Write an expression to print the last row. (It will be a five-element array).

# In[82]:


four_by_five[-1]


# Using the same four_by_five array.Print the 14 value

# In[84]:


four_by_five[2][3]


# Using the same four_by_five array.How could you display the first column? It will be a (four-element array ending with 16.)

# In[95]:


four_by_five[:,0]


# Using the same four_by_five array.Display the last two colums

# In[104]:


four_by_five[:,3:]


# Using the same four_by_five array.Write an expression to return the last two columns of the middle two rows.

# In[105]:


four_by_five[1:3,3:]

