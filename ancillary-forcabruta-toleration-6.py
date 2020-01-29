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
from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister
from qiskit import Aer


from qiskit import BasicAer, execute
from qiskit.tools.visualization import *
from qiskit import IBMQ

from qiskit.tools.visualization import circuit_drawer
from math import sqrt

import pandas as pd
import numpy as np
import random
random.seed(1)

from tqdm import tqdm, trange
import time
from time import sleep

import sklearn
from sklearn.model_selection import train_test_split

backendSimulador = Aer.get_backend('qasm_simulator')


# In[2]:


def returnDictionaryInputs(n):
    """
    this function returns a dict with the keys being the amount of 1s in strings and the value the strings
    """
    dic = {}
    for i in range(2**n):
        state = bin(i)[2:]
        q1 = 0
        for bit in state:
            if bit == '1':
                q1+=1
        if not q1 in dic:
            dic[q1] = [i]
        else:
            dic[q1].append(i)
    
    return dic
            

def returnPositions1InState(s):
    """
    this function returns the positions of 1s in the state s
    """
    binState = bin(s)[2:][::-1]
    #print(binState)
    positions = []
    position=0
    for bit in binState:
        if bit == '1':
            positions.append(position)
        position+=1
    return positions

def returnPositions1InState(s, N):
    """
    this function returns the positions of 1s in the state s with size N
    """
    binState = bin(s)[2:]#[::-1]
    if (len(binState) != N):
        binState = (N - len(binState))*"0" + binState
    #print(binState)
    positions = []
    position=0
    for bit in binState:
        if bit == '1':
            positions.append(position)
        position+=1
    return positions

def checkZmodifyState(positionsOfZ, state, N):
    """
    verify if the state 'state' is modified with application of Z or CZ operations in positions positionsOfZ
    the state 'state' has N size.
    """
    positions1 = returnPositions1InState(state, N)
    for positionOfZ in positionsOfZ:
        if not positionOfZ in positions1:
            return False
    return True
    
def generateCCZ(circ, ctrl, indexCCZ, anc, tgt):
    """
    generate a CCZ in the circuit 'circ' with the ctrl being the controllers register and 'indexCCZ' the positions of 
    the controllers in this register.
    'anc' is the ancillary qubits and 'tgt' the target.
    """
    #indexCCZ = [0,1,2,3,4]
    quantityControllers = len(indexCCZ)
    # compute
    circ.ccx(ctrl[indexCCZ[0]], ctrl[indexCCZ[1]], anc[0])
    for i in range(2, quantityControllers):
        circ.ccx(ctrl[indexCCZ[i]], anc[i-2], anc[i-1])

    # copy
    circ.cz(anc[quantityControllers-2], tgt)

    # uncompute
    for i in range(quantityControllers-1, 1, -1):
        circ.ccx(ctrl[indexCCZ[i]], anc[i-2], anc[i-1])
    circ.ccx(ctrl[indexCCZ[0]], ctrl[indexCCZ[1]], anc[0])    

def generateCCX(circ, ctrl, indexCCZ, anc, tgt):
    """
    generate a CCZ in the circuit 'circ' with the ctrl being the controllers register and 'indexCCZ' the positions of 
    the controllers in this register.
    'anc' is the ancillary qubits and 'tgt' the target.
    """
    #indexCCZ = [0,1,2,3,4]
    quantityControllers = len(indexCCZ)
    # compute
    circ.ccx(ctrl[indexCCZ[0]], ctrl[indexCCZ[1]], anc[0])
    for i in range(2, quantityControllers):
        circ.ccx(ctrl[indexCCZ[i]], anc[i-2], anc[i-1])

    # copy
    circ.cx(anc[quantityControllers-2], tgt)

    # uncompute
    for i in range(quantityControllers-1, 1, -1):
        circ.ccx(ctrl[indexCCZ[i]], anc[i-2], anc[i-1])
    circ.ccx(ctrl[indexCCZ[0]], ctrl[indexCCZ[1]], anc[0])    

def createSuperposition(circuit, Nqubits, qubitsInput):
    #hadamard application in all Nqubits
    for i in range(Nqubits):
        circuit.h(qubitsInput[i])
    

def createXN(circuit, Nqubits, qubitsInput):
    #hadamard application in all Nqubits
    for i in range(Nqubits):
        circuit.x(qubitsInput[i])

def createU(circuit, Nqubits, patterns, patternsInitial, qubitsInput, anc):
    
    #print(patterns)
    if(patterns[0] == -1):
        print("convertendo", patterns)
        patterns = [p*(-1) for p in patterns]
        print("em", patterns)
    
    #print(patterns)    
    dicInputs = returnDictionaryInputs(Nqubits)
    #put Z in the positions with only one 1 in the basis state:
    for state in dicInputs[1]:
        if patterns[state] == -1:
            # print("colocando z nas posicoes 1 no estado", state)
            positions = returnPositions1InState(state,Nqubits)
            for position in positions:
                circuit.z(qubitsInput[position])

                for indexPatternInitial in range(len(patternsInitial)):
                    if checkZmodifyState([position], indexPatternInitial, Nqubits):
                        patternsInitial[indexPatternInitial] *= -1
    for p in range(2,Nqubits+1):
        for state in dicInputs[p]:

            if patterns[state] != patternsInitial[state]: 

                positions = returnPositions1InState(state, Nqubits)
                # print("colocando c-z nas posicoes", positions ,"do circuito devido ao estado", state)

                positionsAux = positions[:len(positions)-1:] #sem a ultima posicao
                indexQubitTarget = positions[len(positions)-1]
                if (len(positions) == 2):
                    circuit.cz(qubitsInput[positions[0]], qubitsInput[positions[1]])
                else:
                    generateCCZ(circuit, qubitsInput, positions, anc, qubitsInput[indexQubitTarget]) 

                for indexPatternInitial in range(len(patternsInitial)):
                    if checkZmodifyState(positions, indexPatternInitial, Nqubits):
                        patternsInitial[indexPatternInitial] *= -1
                        
def printMatrix(matrix):
    """
    print a matrix with size equals to some power of two in a prety way
    """
    for i in range(len(matrix)):
        print(matrix[i], end="    ")
        if (i+1)%sqrt(len(matrix)) == 0 and i!=0:
            print("\n");
            
def printMatrix2(matrix):
    """
    print a matrix with size equals to some power of two in a realy prety way
    """
    for i in range(len(matrix)):
        if matrix[i]==1:
            print('#', end="   ")
        else: 
            print('.', end="   ")
        if (i+1)%sqrt(len(matrix)) == 0 and i!=0:
            print("\n");

def randomMatrix(matrix):
    """
    takes a matrix and rearanges it with random numbers 1 or 0
    """
    for i in range(len(matrix)):
        matrix[i] = random.randrange(2)
    return matrix

def randomRealMatrix(tup):
    rowSize = tup[0]
    columnSize = tup[1]
    matrix = np.empty(shape=(rowSize,columnSize)).astype(int)
    print(matrix.shape)
    for i in range(rowSize):
        for j in range(columnSize):
            matrix[i][j] = random.randrange(2)
        #matrix[i] = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    return matrix

def changeWeight(imat, wmat, flag):
    """
    takes the matrix of i, the matrix of w and a flag representing the training status
    flag = 1 -> the result was 0 but should be 1 and this means that we need to change the componets of w where i and w differ from each other
    flag = 2 -> the result was 1 but should be 0 and this means that we need to change the componets of w where i and w coincide
    this function returns the new Uw changed
    """
    equals = []
    differents = []
    toBeChoosen = []
    choosen = []
    
    for i in range (len(wmat)):
        if wmat[i] == imat[i]: 
            equals.append(i);
        else: differents.append(i);

    # print("Os elementos de i e w que são iguais são " + str(len(equals)) + " e estão nos índices:" + str(equals))
    # print("Os elementos de i e w que são diferentes são " + str(len(differents)) + " e estão nos índices:" + str(differents))

    if flag == 1:toBeChoosen = differents # 0 mas deveria ser 1
    if flag == 2: toBeChoosen = equals # 1 mas deveria ser 0
    
    for i in range (int(ln * len(toBeChoosen))):
        randomIndex = random.randrange(len(toBeChoosen))
        choosen.append(toBeChoosen.pop(randomIndex))

    # print("Como ln é igual a " + str(ln) + ", foram escolhidos aleatoriamente " + str(len(choosen)) + " indexes. São eles: " + str(choosen) )

    for i in range (len(choosen)):
        index = choosen[i]
        if wmat[index] == 0:
            wmat[index] = 1
        else:
            wmat[index] = 0;
    return wmat
   
def transformMatrixToVector(matrix):
    vector=[]
    for elem in matrix:
        if elem == 1:
            vector.append(1)
        else: #i.e == 0
            vector.append(-1)
    # se o primeiro elemento for 0, ele inverte os valores do vetor/matrix       
    if (vector[0] == -1):
        vector = [i*-1 for i in vector]
    return vector

# function for automated Ui or Uw SF generation
''' !!! WARNING: THIS ONLY WORKS WITH 4x4 MATRIX !!! '''
def autoMarkCircuitSF(matrix, circuit, nbase_states):
    ''' !!! WARNING: THIS ONLY WORKS WITH 4x4 MATRIX !!!
        identify number of qubits needed for the base states,
        identify wich states to be marked, 
        create circuit based on state bits
    '''
    # identify positions to be marked with Z-GATE
    circuits_config = []
    for i in range(len(matrix)):
        if matrix[i] == 1:
            state_bits = nbase_states.format(i)
            # print('states marked: ',state_bits)
            # apply NOT to qubits with '0' in position
            for bit_index in range(len(state_bits)): 
                if state_bits[bit_index] == '0':
                    circuit.x(qubitsInput[bit_index])
            # MARK STATE 
            generateCCZ(circuit, qubitsInput, [0,1,2,3], anc, 3)
            # apply NOT to qubits with '0' in position
            for bit_index in range(len(state_bits)): 
                if state_bits[bit_index] == '0':
                    circuit.x(qubitsInput[bit_index])
    return circuit

def increaseThreshold(circuit, tolerationRate, qubitsInput, Nqubits, anc, qubitOutput):
    if(tolerationRate >= 1):
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[3])
    if (tolerationRate >=2):
        circuit.x(qubitsInput[2])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[2])
    if (tolerationRate>=3):
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
    if (tolerationRate>=4):
        circuit.x(qubitsInput[1])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[1]) 
    if (tolerationRate>=5):
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[3])
    if (tolerationRate>=6):
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
    
def runHypergraph (weightsV, dataset, nshots, threshold, toleration):
    """
    This function takes
    weightsV = an initial array of weights only to test it, not to improve
    dataset = the dataset to split it in to the array of input and the class for all the dataset
    nshots = the number of shots to run on qasm_simulator
    threshold = the % of nshots that we should consider that 1 is the winner
    """
    acertos = 0

    erros = 0
    
    lenInput = len(dataset[0][0:-1])
    
    patternsInitialI = [1  for i in range(lenInput)]

    patternsInitialW = [1  for i in range(lenInput)]

    tempoTotal = 0

    random.seed(1)

    '''Inicializando/Resetando o circuito
    '''
    Nqubits = int(sqrt(lenInput))
    qubitsInput = QuantumRegister(Nqubits, 'qin')
    anc = QuantumRegister(Nqubits-1, 'anc')
    qubitOutput = QuantumRegister(1, 'qout')
    output = ClassicalRegister(1, 'cout')

    '''Testando weightsV em todo o dataset
    '''
    program_starts = time.time()
    for i in range(datasize):

        # Vetor de entrada Ui (estados a serem marcados)
        inputV =  dataset[i][0:-1]

        # valor da class do Ui atual (é cruz o não?)
        theClass = dataset[i][-1]
        circuit = QuantumCircuit(qubitsInput,anc,qubitOutput,output)

        #input
        patternsI = transformMatrixToVector(inputV)
        #weights
        patternsW = transformMatrixToVector(weightsV)
        def produtoInterno(x,w):
            tamX = len(x)
            prodInterno = 0
            for i in range(tamX):
                prodInterno += x[i]*w[i]
            return prodInterno
        def perceptron(x,w, limiar):
            prodInterno = produtoInterno(x,w)
            if(prodInterno >= limiar):
                return 1
            else:
                return 0
        #print("entrada",inputV)
        #print("peso", weightsV)
        #print("peso convertido",patternsW)
        #saidaClassica = perceptron(inputV, patternsW, 0.5)
        #print("saida esperada",theClass)
        #print("saida classica",saidaClassica)
        
        createSuperposition(circuit, Nqubits, qubitsInput) # Superposiçóes inicial
        patternsInitialI = [1  for i in range(lenInput)]
        patternsInitialW = [1  for i in range(lenInput)]
        createU(circuit, Nqubits, patternsI, patternsInitialI, qubitsInput, anc) # Ui

        circuit.barrier()

        createU(circuit, Nqubits, patternsW, patternsInitialW, qubitsInput, anc) # Uw
        createSuperposition(circuit, Nqubits, qubitsInput) # Superposiçóes final
        createXN(circuit, Nqubits, qubitsInput) # Nots finais
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput) # Produto interno
        
        increaseThreshold(circuit, toleration, qubitsInput, Nqubits, anc, qubitOutput  )


        circuit.measure(qubitOutput, output) # Medição 

        circuit.barrier()

        job = execute(circuit, backend=backendSimulador, shots=nshots)
        result = job.result()
        contagem = result.get_counts()

        results1 = contagem.get('1') # Resultados que deram 1
        if str(type(results1)) == "<class 'NoneType'>": results1 = 0
        results0 = contagem.get('0') # Resultados que deram 0
        if str(type(results0)) == "<class 'NoneType'>": results0 = 0

        # Utilizando threshold
        if results1 >= (nshots * threshold):
            ##print("saida quantico 1")
            its1 = True
            its0 = False
        else:
            ##print("saida quantico 0")
            its1 = False
            its0 = True
        ##print("results 0", results0, "results 1",results1 )
        
        # Verificação de resultado e classe: acerto/erro
        if its0: #neuronio deu como saida 0
            if theClass != 0: 
                erros = erros + 1
            else:
                acertos = acertos + 1

        else: #neuronio deu como saida 1
            if theClass != 1: 
                erros = erros + 1
            else:
                acertos = acertos + 1

    now = time.time()
    tempoTotal = now - program_starts

#     print("Acurácia: ", "{0:.2f}".format((acertos/datasize) * 100),"%| Erros: ", "{0:.2f}".format((erros/datasize) * 100), "% | Tempo: ", "%.2f" % tempoTotal, " segundos")
#     print(weightsV, float("{0:.2f}".format((acertos/datasize) * 100)))
    return float("{0:.2f}".format((acertos/datasize) * 100))
def fitness(weights, scores):
    '''
    Sorts a with respect to b and sorts b too
    In the end the indexes of a and b will still be corresponded to each other
    '''
    weights = np.array(weights)
    scores = np.array(scores)
    idx_sorted = scores.argsort()
    
    return (weights[idx_sorted][::-1], scores[idx_sorted][::-1])

def generationOut(population, dataset, nshots, threshold, toleration):
#     print("Calculating the fitness...")
    ''' Takes some population and returns the population and the acuracy of each cromossom in that dataset sorted side by side
    '''
    acuracy = np.zeros(len(population))
    for i in range(len(population)):
        acuracy[i] = runHypergraph (weightsV = population[i],
                                    dataset = dataset,
                                    nshots = nshots,
                                    threshold = threshold,
                                    toleration=toleration)
    
    return fitness(population, acuracy)


# ### Inicialização Básica

# In[3]:


# Inicialização da base de dados
df = pd.read_csv("data/data9-6-menos1.csv")

df = sklearn.utils.shuffle(df) 

dataset = df.values.tolist()

lenInput = len(dataset[0][0:-1])

datasize = len(dataset)

print("Base com", datasize, "exemplares")
print("Cada exemplar com", lenInput, "valores além da classe")


# ### Treinamento - Hipergrafos

# In[4]:


def makeCrossover(weight1, weight2):
#     print("Making Crossover")
    '''
        WARNING!!! This will only work with even weight's size 
    '''
    hwsize = int(len(weight1)/2)
    
    head1 = weight1[:hwsize]
    #tail1 = weight1[hwsize:]
    #head2 = weight2[:hwsize]
    tail2 = weight2[hwsize:]
    child1 = np.concatenate((head1, tail2), axis=0)
    #child2 = np.concatenate((head2, tail1), axis=0)

    return child1#(child1, child2)

def rouletteAlgo(ranks):
    sumWeights = sum (ranks)
    sumWeightsNormalized = [i/sumWeights for i in ranks]
    #print(sumWeightsNormalized)
    randomUntilSum = random.random()
    #print(randomUntilSum)
    chosenPosition = -1;
    sumAux=0
    while(sumAux <= randomUntilSum):
        chosenPosition+=1;
        sumAux += sumWeightsNormalized[chosenPosition];
    return chosenPosition
    
def makeNewGen(weights, ranks, qNews):
#     print("Making new generation...")
    childs = np.array([np.array([2  for i in range(len(weights[0]))]) for i in range(qNews)]) # inicializando um array bidimensional da forma mais burra possivel
    #hwsize = int(len(weights)/2)
    i=0
    
    while(i < qNews):
        child = makeCrossover(weights[rouletteAlgo(ranks)], weights[rouletteAlgo(ranks)])
        childs[i] = np.array(child)
        i +=1
    return makeMutation(childs)

def makeMutation(weights):
#     print("Making Mutation...")
    print(weights)
    for weightIndex in range(len(weights)):
        for geneIndex in range(len(weights[weightIndex])):
            randomValue = random.random()
            if(randomValue <= mutationRate):
                weights[weightIndex][geneIndex] = int (not weights[weightIndex][geneIndex])
    return weights       

    #numberOfMutations = (len(weights[0]) * len(weights)) / 100 # 1% dos genes
    #whichOnes = [0 for i in range(round(numberOfMutations))] # indice dos cromossomos que vão sofrer mutações
    
    #for i in range(round(numberOfMutations)):
    #    whichOnes[i] = random.randrange(len(weights))
    
    #for i in range(len(whichOnes)):
    #    
    #    j = random.randrange(len(weights[0])) # indice do gene que vai sofrer mutação agora
    #    
    #    if weights[whichOnes[i]][j] == 0:
    #        weights[whichOnes[i]][j] = 1
    #    else: weights[whichOnes[i]][j] = 0
        
#         print("Mutacoes nos indices: ", whichOnes[i], j)
    
    #return weights


# In[5]:


mutationRate=0.1
sol_per_pop = 5 # How many chromosome there is in that population
toleration=4
pop_size = (sol_per_pop, lenInput) # The population will have sol_per_pop chromosome where eachchromosome has lenInput genes


# In[6]:


new_population = randomRealMatrix(pop_size)

new_population


# In[7]:


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


def forcaBruta(nshots, threshold, toleration):
    bestAcc=0
    bestWeight=0
    for i in range(2**16):
        program_starts = time.time()
        
        weight = inteiroParaPeso(i,16)
        accuracyNow = runHypergraph (weightsV = weight,
                                         dataset = dataset,
                                         nshots = nshots,
                                         threshold = threshold,
                                        toleration=toleration)
        if(accuracyNow > bestAcc):
            bestAcc = accuracyNow
            bestWeight = i
            
        now = time.time()
        tempoTotal = now - program_starts
        #print("Melhor peso: ", theBestWeight, "Acuracia:", theBestAccuracy, "Tempo:", tempoTotal, "segundos")
        print("iteration", i, "theBestAccuracy, ", bestAcc, "best weight",  bestWeight, "Tempo:", "{0:.2f}".format(tempoTotal), "segundos")


    return (bestWeight, bestAcc)


# In[ ]:


forcaBruta(nshots=1000, threshold=0.5, toleration=6)


# In[ ]:





# In[ ]:





# In[ ]:




