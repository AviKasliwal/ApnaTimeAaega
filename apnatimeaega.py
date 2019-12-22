import numpy as np
np.random.seed(seed = 11)

def generateTable(data,k=4):
    
    T = {}
    for i in range(len(data)-k):
        X = data[i:i+k]
        Y = data[i+k]
        #print("X  %s and Y %s  "%(X,Y))
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T


def convertFreqIntoProb(T):     
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
                
    return T

text_path = "Apna Time Aayega.txt"
def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text = load_text(text_path)

# print(text[:1000])

def trainMarkovChain(text,k=4):
    
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    
    return T

model = trainMarkovChain(text)

def sample_next(ctx,T,k):
    ctx = ctx[-k:]
    if T.get(ctx) is None:
        return " "
    possible_Chars = list(T[ctx].keys())
    possible_values = list(T[ctx].values())
    
#     np.random.seed(seed = 11)
    return np.random.choice(possible_Chars,p=possible_values)



def generateText(starting_sent,k=4,maxLen=1000):
    
    sentence = starting_sent
    ctx = starting_sent[-k:]
    
    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence

text = generateText("apna",k=4,maxLen=2000)

file1 = open("seed1.txt","w")
file1.write(text) 
file1.close()
