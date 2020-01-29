#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

def produtoInterno(x,w):
    tamX = len(x)
    prodInterno = 0
    for i in range(tamX):
        prodInterno += x[i]*w[i]
    return prodInterno

def qIguais(x1,x2):
    cont=0
    for i in range(len(x1)):
        if x1[i] == x2[i]:
            cont+=1
    return cont

def perceptron(x,w, limiar):
    prodInterno = produtoInterno(x,w)
    if(prodInterno >= limiar):
        return 1
    else:
        return 0

def perceptronContador(x,w, limiar):
    prodInterno = qIguais(x,w)
    if(prodInterno >= limiar):
        return 1
    else:
        return 0
    
def inteiroParaPeso(inteiro, qBits):
    stringNumeroBinario = bin(inteiro)[2:]
    pesoString = (qBits-len(stringNumeroBinario))*"0"+stringNumeroBinario
    peso = []
    for c in pesoString:
        if(c == '1'):
            peso.append(1)
        else:
            peso.append(-1)
    return peso

def inteiroParaPeso01(inteiro, qBits):
    stringNumeroBinario = bin(inteiro)[2:]
    pesoString = (qBits-len(stringNumeroBinario))*"0"+stringNumeroBinario
    peso = []
    for c in pesoString:
        if(c == '1'):
            peso.append(1)
        else:
            peso.append(0)
    return peso


# In[32]:


# Inicialização da base de dados
df = pd.read_csv("data/data9-6.csv")

df = sklearn.utils.shuffle(df) 

dataset = df.values.tolist()

lenInput = len(dataset[0][0:-1])

datasize = len(dataset)

print("Base com", datasize, "exemplares")
print("Cada exemplar com", lenInput, "valores além da classe")


# In[34]:


maxAcerto=0
pesoDoMax=0
for pesoInteiro in range(2**16):
    pesoVetor = inteiroParaPeso(pesoInteiro, 16)
    acertos=0
    erros=0
    for i in range(datasize):
        inputV =  dataset[i][0:-1]
        theClass = dataset[i][-1]
        saida = perceptronContador(inputV, pesoVetor, 7)
        if (theClass == saida):
            acertos+=1
        else:
            erros+=1
    if(acertos>maxAcerto):
        maxAcerto = acertos
        pesoDoMax = pesoInteiro
    print("peso", pesoInteiro, "acertos",acertos, "max", maxAcerto, "pesoMax", pesoDoMax)
        


# In[35]:


maxAcerto=0
pesoDoMax=0
for pesoInteiro in range(2**16):
    pesoVetor = inteiroParaPeso(pesoInteiro, 16)
    acertos=0
    erros=0
    for i in range(datasize):
        inputV =  dataset[i][0:-1]
        theClass = dataset[i][-1]
        saida = perceptronContador(inputV, pesoVetor, 6)
        if (theClass == saida):
            acertos+=1
        else:
            erros+=1
    if(acertos>maxAcerto):
        maxAcerto = acertos
        pesoDoMax = pesoInteiro
    print("peso", pesoInteiro, "acertos",acertos, "max", maxAcerto, "pesoMax", pesoDoMax)
        


# In[36]:


maxAcerto=0
pesoDoMax=0
for pesoInteiro in range(2**16):
    pesoVetor = inteiroParaPeso(pesoInteiro, 16)
    acertos=0
    erros=0
    for i in range(datasize):
        inputV =  dataset[i][0:-1]
        theClass = dataset[i][-1]
        saida = perceptronContador(inputV, pesoVetor, 5)
        if (theClass == saida):
            acertos+=1
        else:
            erros+=1
    if(acertos>maxAcerto):
        maxAcerto = acertos
        pesoDoMax = pesoInteiro
    print("peso", pesoInteiro, "acertos",acertos, "max", maxAcerto, "pesoMax", pesoDoMax)
        


# In[ ]:


maxAcerto=0
pesoDoMax=0
for pesoInteiro in [60030]:
    pesoVetor = inteiroParaPeso(pesoInteiro, 16)
    acertos=0
    erros=0
    for i in range(datasize):
        inputV =  dataset[i][0:-1]
        theClass = dataset[i][-1]
        print(inputV)
        print(pesoVetor)
        saida = perceptron(inputV, pesoVetor, 0.5)
        print(saida, theClass)
        if (theClass == saida):
            acertos+=1
        else:
            erros+=1
    if(acertos>maxAcerto):
        maxAcerto = acertos
        pesoDoMax = pesoInteiro
    print("peso", pesoInteiro, "acertos",acertos, "max", maxAcerto, "pesoMax", pesoDoMax)
        


# In[ ]:





# In[ ]:





# In[ ]:




