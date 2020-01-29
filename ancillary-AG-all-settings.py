#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
implementacao do neuronio quantico definido 
no artigo

'An artificial neuron implemented on an actual quantum
processor' by (Francesco Tacchino, Chiara Macchiavello, Dario Gerace and Daniele Bajoni)

código escrito por Fernando M de Paula Neto e bagunçado por Gustavo I dos Santos F

"""
from functions import *


# ### Treinando com AG os pesos do circuito e o nivel de tolerância

# In[ ]:


GALearning(databaseName="data/data9-6-menos1.csv", 
           generations= 200, 
           nshots=8192, 
           threshold=0.5, 
           toleration=0, 
           mutationRate=0.1, 
           sol_per_pop=10,
           printPopulation=True,
           qBias=4,
           qBiasUsing=4,
           toTrainToleration=True)


# ### Treinando com AG os pesos do circuito com um determinado/fixo nivel de tolerancia:

# In[ ]:


GALearning(databaseName="data/data9-6-menos1.csv", 
           generations= 200, 
           nshots=8192, 
           threshold=0.5, 
           toleration=6, 
           mutationRate=0.1, 
           sol_per_pop=10,
           printPopulation=True,
           qBias=0,
           qBiasUsing=0,
           toTrainToleration=False)


# ### Treinando com AG os pesos e bias do circuito com um determinado/fixo nivel de tolerancia

# In[ ]:


GALearning(databaseName="data/data9-6-menos1.csv", 
           generations= 200, 
           nshots=8192, 
           threshold=0.5, 
           toleration=6, 
           mutationRate=0.1, 
           sol_per_pop=10,
           printPopulation=True,
           qBias=16,
           qBiasUsing=8,
           toTrainToleration=False)


# In[ ]:




