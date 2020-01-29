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
import math

"""
from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister
from qiskit import Aer


from qiskit import BasicAer, execute
from qiskit.tools.visualization import *
from qiskit import IBMQ

# from qiskit.tools.visualization import circuit_drawer
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

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np



"""
backendSimulador = Aer.get_backend('qasm_simulator')


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

def createU(circuit, Nqubits, patterns, qubitsInput, anc):
    
    patternsInitial = [1  for i in range(2**Nqubits)]

    #print(patterns)
    if(patterns[0] == -1):
#         print("convertendo", patterns)
        patterns = [p*(-1) for p in patterns]
#         print("em", patterns)
    
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
            if (random.randrange(2)==0):
                matrix[i][j] = 1
            else:
                matrix[i][j] = -1
        #matrix[i] = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    return matrix

def randomRealMatrixWithBias(tup, qBias):
    rowSize = tup[0]
    columnSize = tup[1]
    matrix = np.empty(shape=(rowSize,columnSize+qBias)).astype(int)
    print(matrix.shape)
    for i in range(rowSize):
        for j in range(columnSize+qBias):
            if (random.randrange(2)==0):
                matrix[i][j] = 1
            else:
                matrix[i][j] = -1
        #matrix[i] = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    return matrix

def randomRealMatrixWithBias(tup, qBias, qBiasUsing):
    rowSize = tup[0]
    columnSize = tup[1]
    matrix = np.empty(shape=(rowSize,columnSize+qBias)).astype(int)
    print(matrix.shape)
    for i in range(rowSize):
        for j in range(columnSize+qBias):
            if (j >= columnSize + qBiasUsing + (qBias-qBiasUsing)//2 ):
              matrix[i][j] = -1
            elif (j >= columnSize + qBiasUsing ):
              matrix[i][j] = +1
            else:
              if (random.randrange(2)==0):
                  matrix[i][j] = 1
              else:
                  matrix[i][j] = -1
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
    if (tolerationRate>=7):
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
    if (tolerationRate >=8):
        circuit.x(qubitsInput[0])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
    if (tolerationRate >=9):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[3])
    if (tolerationRate >=10):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[2])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[2])
    if (tolerationRate >=11):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
    if (tolerationRate >=12):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
    if (tolerationRate >=13):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[3])
    if (tolerationRate >=14):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2]) 
    if (tolerationRate >=15):
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
        generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)
        circuit.x(qubitsInput[0])
        circuit.x(qubitsInput[1])
        circuit.x(qubitsInput[2])
        circuit.x(qubitsInput[3])
             

    
    
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
        saidaClassica = perceptron(inputV, patternsW, 0.5)
        #print("saida esperada",theClass)
        #print("saida classica",saidaClassica)
        
        createSuperposition(circuit, Nqubits, qubitsInput) # Superposiçóes inicial
        
        
        createU(circuit, Nqubits, patternsI, qubitsInput, anc) # Ui

        circuit.barrier()

        createU(circuit, Nqubits, patternsW, qubitsInput, anc) # Uw
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


def executeNeuronReturnCount(i,w, toleration, nShots):
    #backendSimulador = BasicAer.get_backend('qasm_simulator')
    backendSimulador = Aer.get_backend('qasm_simulator')
    Nqubits = int(math.log(len(i),2))
    qubitsInput = QuantumRegister(Nqubits, 'qin')
    anc = QuantumRegister(Nqubits-1, 'anc')
    qubitOutput = QuantumRegister(1, 'qout')
    output = ClassicalRegister(1, 'cout')

    circuit = QuantumCircuit(qubitsInput,anc,qubitOutput,output)

    patterns = i #[1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]#[1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1]#[-1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1]#[1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1]#[-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    createSuperposition(circuit, Nqubits, qubitsInput)
    createU(circuit, Nqubits, patterns, qubitsInput, anc)
    circuit.barrier()

    #weights
    patterns = w #[1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1]#[1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1]#[1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1]

    createU(circuit, Nqubits, patterns, qubitsInput, anc)
    createSuperposition(circuit, Nqubits, qubitsInput)
    createXN(circuit, Nqubits, qubitsInput)
    generateCCX(circuit, qubitsInput, range(Nqubits), anc, qubitOutput)

    increaseThreshold(circuit, toleration, qubitsInput, Nqubits, anc, qubitOutput  )

    circuit.measure(qubitOutput, output)
    circuit.barrier()
    #circuit_drawer(circuit)
    #circuit.draw(output='mpl')

    job = execute(circuit, backend=backendSimulador, shots=nShots) #8192
    result = job.result()
    count = result.get_counts()
    
    return count


def runHypergraphWithBiasWithNeuron (weightsV, dataset, nshots, threshold, toleration, qBias):
    """
    This function takes
    weightsV = an initial array of weights only to test it, not to improve
    dataset = the dataset to split it in to the array of input and the class for all the dataset
    nshots = the number of shots to run on qasm_simulator
    threshold = the % of nshots that we should consider that 1 is the winner
    """
    
    acertos = 0

    erros = 0
    
    lenInput = len(dataset[0][0:-1]) + qBias
    
    datasize = len(dataset)

    tempoTotal = 0

    random.seed(1)

   

    '''Testando weightsV em todo o dataset
    '''
    program_starts = time.time()
    for i in range(datasize):

        # Vetor de entrada Ui (estados a serem marcados)
        inputV =  dataset[i][0:-1] + [1 for iAux in range(qBias)]

        # valor da class do Ui atual (é cruz o não?)
        theClass = dataset[i][-1]
        
        #input
        patternsI = transformMatrixToVector(inputV)
        #weights
        patternsW = transformMatrixToVector(weightsV)
       
        contagem = executeNeuronReturnCount(patternsI, patternsW, toleration, nshots)

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


def runHypergraphWithBias (weightsV, dataset, nshots, threshold, toleration, qBias):
    """
    This function takes
    weightsV = an initial array of weights only to test it, not to improve
    dataset = the dataset to split it in to the array of input and the class for all the dataset
    nshots = the number of shots to run on qasm_simulator
    threshold = the % of nshots that we should consider that 1 is the winner
    """
    
    acertos = 0

    erros = 0
    
    lenInput = len(dataset[0][0:-1]) + qBias
    
    datasize = len(dataset)

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
        inputV =  dataset[i][0:-1] + [1 for iAux in range(qBias)]

        # valor da class do Ui atual (é cruz o não?)
        theClass = dataset[i][-1]
        circuit = QuantumCircuit(qubitsInput,anc,qubitOutput,output)

        #input
        patternsI = transformMatrixToVector(inputV)
        #weights
        patternsW = transformMatrixToVector(weightsV)
        
        createSuperposition(circuit, Nqubits, qubitsInput) # Superposiçóes inicial
        patternsInitialI = [1  for i in range(lenInput)]
        patternsInitialW = [1  for i in range(lenInput)]
        createU(circuit, Nqubits, patternsI, qubitsInput, anc) # Ui

        circuit.barrier()

        createU(circuit, Nqubits, patternsW, qubitsInput, anc) # Uw
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

def generationOut(population, dataset, nshots, threshold, toleration, qBias, qToleration):
#     print("Calculating the fitness...")
    ''' Takes some population and returns the population and the acuracy of each cromossom in that dataset sorted side by side
    '''
    acuracy = np.zeros(len(population))
    for i in range(len(population)):
        if (qToleration):
            weight = population[i][:len(population[i]) - qBias]
            toleration = binToInt(population[i][len(population[i]) - qBias:])

            acuracy[i] = runHypergraphWithBiasWithNeuron (weightsV = weight,
                                    dataset = dataset,
                                    nshots = nshots,
                                    threshold = threshold,
                                    toleration=toleration,
                                    qBias = 0)

        else:
            weight = population[i]
            acuracy[i] = runHypergraphWithBiasWithNeuron (weightsV = weight,
                                    dataset = dataset,
                                    nshots = nshots,
                                    threshold = threshold,
                                    toleration=toleration,
                                    qBias = qBias)
        
        
    
    return fitness(population, acuracy)

def convertTime(seconds):
    ''' Transforms some seconds amount in to the format hh:mm:ss
    '''
    hours = 0
    minutes = 0
    
    if seconds > 60:
        minutes = int(seconds/60)
        seconds = int(seconds%60)
    
    if minutes > 60:
        hours = int(minutes/60)
        minutes = int(hours%60)
    
    if minutes < 10: minutes = "0" + str(minutes)
    if seconds < 10: seconds = "0" + str(seconds)
    if hours < 10: hours = "0" + str(hours)
    
    return ("[" + str(hours) + ":" + str(minutes) + ":" + str(seconds) + "]")



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


def makeCrossoverWithBias(weight1, weight2, qBias, qBiasUsing):
#     print("Making Crossover")
    '''
        WARNING!!! This will only work with even weight's size 
    '''
    hwsize = int((len(weight1)-qBias)/2)
    #print(hwsize)
    head1 = weight1[:hwsize]
    #tail1 = weight1[hwsize:]
    #head2 = weight2[:hwsize]
    tail2 = weight2[hwsize:2*hwsize]
    
    head1_t = weight1[2*hwsize:2*hwsize + qBiasUsing//2]
    tail2_t = weight2[2*hwsize + qBiasUsing//2:]
    #print(head1, tail2, head1_t, tail2_t)
    child1 = np.concatenate((head1, tail2, head1_t, tail2_t ), axis=0)
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
    
def makeNewGen(weights, ranks, qNews, qBias, qBiasUsing, mutationRate):
#     print("Making new generation...")
    childs = np.array([np.array([2  for i in range(len(weights[0]))]) for i in range(qNews)]) # inicializando um array bidimensional da forma mais burra possivel
    #hwsize = int(len(weights)/2)
    i=0
    
    while(i < qNews):
        child = makeCrossoverWithBias(weights[rouletteAlgo(ranks)], weights[rouletteAlgo(ranks)], qBias, qBiasUsing)
        childs[i] = np.array(child)
        i +=1
    return makeMutationBias(childs, qBias, qBiasUsing, mutationRate)



def makeMutation(weights):
    for weightIndex in range(len(weights)):
        for geneIndex in range(len(weights[weightIndex])):
            randomValue = random.random()
            if(randomValue <= mutationRate):
                weights[weightIndex][geneIndex] = (-1)*weights[weightIndex][geneIndex]
    return weights       

def makeMutationBias(weights, qBias, qBiasUsing, mutationRate):
    for weightIndex in range(len(weights)):
        for geneIndex in range(len(weights[weightIndex])):
            randomValue = random.random()
            if(randomValue <= mutationRate and geneIndex < len(weights[0])-(qBias - qBiasUsing)):
                weights[weightIndex][geneIndex] = (-1)*weights[weightIndex][geneIndex]
    return weights      


def GALearningFromPopulation(database, generations, nshots, threshold, toleration, mutationRate, sol_per_pop,printPopulation, qBias, qBiasUsing, toTrainToleration, population, continueFrom):

    
    if (toTrainToleration):
        print("WARNING: GA will train toleration parameter.")
        print("Put in the qBias/qBiasUsing the amount of qubits used to represent the toleration parameter")

    # Inicialização da base de dados
    df = database
    df = sklearn.utils.shuffle(df) 
    dataset = df.values.tolist()
    lenInput = len(dataset[0][0:-1])
    datasize = len(dataset)

    print("Base com", datasize, "exemplares")
    print("Cada exemplar com", lenInput, "valores além da classe")
    #mutationRate=0.1
    #sol_per_pop = 50 # How many chromosome there is in that population
    # toleration=6
    pop_size = (sol_per_pop, lenInput) # The population will have sol_per_pop chromosome where eachchromosome has lenInput genes

    # new_population = randomRealMatrixWithBias(pop_size,qBias = qBias, qBiasUsing=qBiasUsing)

    # population = new_population

    accuracyList = [0  for i in range(generations)]
    # Ranqueando população com base na acurácia
    rankedPopulation, rank = generationOut(population = population,
                                          dataset = dataset,
                                          nshots = nshots,
                                          threshold = threshold,
                                          toleration=toleration,
                                          qBias = qBias,
                                          qToleration = toTrainToleration)
    #print("populacao rankeada")
    #print(rankedPopulation)
    
    #print("rank")
    #print(rank)
    #print("")
    for i in range(generations):
        print("Generation", continueFrom+i+1)
        theBestWeight = rankedPopulation[0]
        theBestAccuracy = rank[0]
        accuracyList[i] = theBestAccuracy

        if(toTrainToleration):
            theBestWeight_PRINT = theBestWeight[:len(theBestWeight) - qBias]
            toleration_PRINT = binToInt(theBestWeight[len(theBestWeight) - qBias:])
            theBestAccuracy = rank[0]
            accuracyList[i] = theBestAccuracy
            print("Best Weight: ", theBestWeight_PRINT, "Toleration", toleration_PRINT, "Accuracy:", rank[0])
            print("----------------")  
            print("populacao rankeada")
            print(rankedPopulation)
            
            print("rank")
            print(rank)
            print("----------------")  
        else:    
            print("Best Weight: ", rankedPopulation[0], "Accuracy:", rank[0])

            print("----------------")  
            print("populacao rankeada")
            print(rankedPopulation)
            
            print("rank")
            print(rank)
            print("----------------")  

        x = np.arange(len(accuracyList))
        
        
        program_starts = time.time()
        
        population = makeNewGen(weights = rankedPopulation, ranks=rank, qNews = sol_per_pop-1, qBias=qBias, qBiasUsing=qBiasUsing, mutationRate=mutationRate)
        
        theBestWeight = rankedPopulation[0]
        population = np.concatenate((population, [theBestWeight]), axis=0) # pegando os 4 melhores pesos e concatenando com seus filhos

        rankedPopulation, rank = generationOut(population = population,
                                              dataset = dataset,
                                              nshots = nshots,
                                              threshold = threshold,
                                              toleration=toleration,
                                              qBias = qBias,
                                              qToleration = toTrainToleration)

        #theBestWeight = rankedPopulation[0]
        #theBestAccuracy = rank[0]
        
        #print("populacao rankeada")
        #print(rankedPopulation)

        #print("rank")
        #print(rank)
        #print("")
        
        now = time.time()
        tempoTotal = now - program_starts
        print("Time:", convertTime(tempoTotal))
    
    fig, ax = plt.subplots()
    plt.bar(x, accuracyList)
    plt.xticks(x, tuple([i  for i in range(generations)]))
    plt.show()
    
    return (theBestWeight, theBestAccuracy)

def GALearning(database, generations, nshots, threshold, toleration, mutationRate, sol_per_pop,printPopulation, qBias, qBiasUsing, toTrainToleration):

    
    if (toTrainToleration):
        print("WARNING: GA will train toleration parameter.")
        print("Put in the qBias/qBiasUsing the amount of qubits used to represent the toleration parameter")

    # Inicialização da base de dados
    df = database
    df = sklearn.utils.shuffle(df) 
    dataset = df.values.tolist()
    lenInput = len(dataset[0][0:-1])
    datasize = len(dataset)

    print("Base com", datasize, "exemplares")
    print("Cada exemplar com", lenInput, "valores além da classe")
    #mutationRate=0.1
    #sol_per_pop = 50 # How many chromosome there is in that population
    # toleration=6
    pop_size = (sol_per_pop, lenInput) # The population will have sol_per_pop chromosome where eachchromosome has lenInput genes

    new_population = randomRealMatrixWithBias(pop_size,qBias = qBias, qBiasUsing=qBiasUsing)

    population = new_population

    accuracyList = [0  for i in range(generations)]
    # Ranqueando população com base na acurácia
    rankedPopulation, rank = generationOut(population = population,
                                          dataset = dataset,
                                          nshots = nshots,
                                          threshold = threshold,
                                          toleration=toleration,
                                          qBias = qBias,
                                          qToleration = toTrainToleration)
    #print("populacao rankeada")
    #print(rankedPopulation)
    
    #print("rank")
    #print(rank)
    #print("")
    for i in range(generations):
        print("Generation", i+1)
        theBestWeight = rankedPopulation[0]
        theBestAccuracy = rank[0]
        accuracyList[i] = theBestAccuracy

        if(toTrainToleration):
            theBestWeight_PRINT = theBestWeight[:len(theBestWeight) - qBias]
            toleration_PRINT = binToInt(theBestWeight[len(theBestWeight) - qBias:])
            theBestAccuracy = rank[0]
            accuracyList[i] = theBestAccuracy
            print("Best Weight: ", theBestWeight_PRINT, "Toleration", toleration_PRINT, "Accuracy:", rank[0])
            print("----------------")  
            print("populacao rankeada")
            print(rankedPopulation)
            
            print("rank")
            print(rank)
            print("----------------")  
        else:    
            print("Best Weight: ", rankedPopulation[0], "Accuracy:", rank[0])

            print("----------------")  
            print("populacao rankeada")
            print(rankedPopulation)
            
            print("rank")
            print(rank)
            print("----------------")  

        x = np.arange(len(accuracyList))
        
        
        program_starts = time.time()
        
        population = makeNewGen(weights = rankedPopulation, ranks=rank, qNews = sol_per_pop-1, qBias=qBias, qBiasUsing=qBiasUsing, mutationRate=mutationRate)
        
        theBestWeight = rankedPopulation[0]
        population = np.concatenate((population, [theBestWeight]), axis=0) # pegando os 4 melhores pesos e concatenando com seus filhos

        rankedPopulation, rank = generationOut(population = population,
                                              dataset = dataset,
                                              nshots = nshots,
                                              threshold = threshold,
                                              toleration=toleration,
                                              qBias = qBias,
                                              qToleration = toTrainToleration)

        #theBestWeight = rankedPopulation[0]
        #theBestAccuracy = rank[0]
        
        #print("populacao rankeada")
        #print(rankedPopulation)

        #print("rank")
        #print(rank)
        #print("")
        
        now = time.time()
        tempoTotal = now - program_starts
        print("Time:", convertTime(tempoTotal))
    
    fig, ax = plt.subplots()
    plt.bar(x, accuracyList)
    plt.xticks(x, tuple([i  for i in range(generations)]))
    plt.show()
    
    return (theBestWeight, theBestAccuracy)


def binToInt(binList):
    n = 0
    position = 0
    for i in binList[::-1]:
        if(i==1):
            n+= 2**(position)
        else:
            n+= 0 * (2**(position))
        position+=1
    return n
