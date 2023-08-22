import numpy as np
from copy import deepcopy
from Engine import Board,bini,index
import matplotlib.pyplot as plt
import tkinter as tk


from PIL import ImageGrab

def getter(root,name):
    x=root.winfo_rootx()
    y=root.winfo_rooty()
    x1=root.winfo_rootx()+240
    y1=root.winfo_rooty()+240
    ImageGrab.grab().crop((x,y,x1,y1)).save(name)


class Layer_Dense: #hidden layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        #minus the largest value from each output to stop large exp values
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Brain:
    def __init__(self):
        self.lay1 = Layer_Dense(24,8)
        self.act1 = Activation_ReLU()
        self.lay2 = Layer_Dense(8,4)
        self.act2 = Activation_Softmax()
    def forward(self, inputs):
       
        self.lay1.forward(inputs)
        self.act1.forward(self.lay1.output)
        
        self.lay2.forward(self.act1.output)
        self.act2.forward(self.lay2.output)

        self.output = self.act2.output

def multi_mutate(brains, x): #keeps the brains and makes x many children asexually
    out = brains

    for i in range(len(brains)):
        brain = brains[i]
        for i in range(x):
            new_brain = deepcopy(brain)

            new_brain.lay1.weights += np.random.randn(24, 8) * np.random.randn(24, 8)
            new_brain.lay1.biases += 0.01 * np.random.randn(1,8)
            new_brain.lay2.weights += np.random.randn(8, 4) * np.random.randn(8, 4)
            new_brain.lay2.biases += 0.01 * np.random.randn(1,4)
        
            out.append(new_brain)

    return out

def learn(networks,gens):

    top_brains = int(networks/10) #top 10% stay alive in each gen
    trial_brains = [Brain() for i in range(networks)]
    output = []

    for i in range(gens):
        scores = []
        
        for brain in trial_brains: #THIS IS WHERE YOUR SCORING / LOSS FUNCTION GOES
            score = test(brain) #score each brain
            scores.append(score)

        print('Gen :',i)
        print('Best Score :',max(scores))
        print('Avg Score :',sum(scores) / len(scores))

        max_index = scores.index(max(scores))
        best_brain = trial_brains[max_index]

        output.append([scores,best_brain])

        scores_copy = scores.copy()
        multi_best_brains = [] #list for best brains
        for x in range(top_brains): #picks out top 10% brains
            multi_best_brains.append(trial_brains[scores_copy.index(max(scores_copy))])

            scores_copy.remove(max(scores_copy))


        trial_brains = multi_mutate(multi_best_brains, 9) #mutate

        plot_data(output)

    return output


def save_brain(brain,filename):
    out = np.array([brain.lay1.weights,
                    brain.lay1.biases,
                    brain.lay2.weights,
                    brain.lay2.biases
                    ])

    np.save(filename,out,)

def open_brain(filename):
    out = np.load(filename,allow_pickle=True)

    brain = Brain()
    brain.lay1.weights = out[0]
    brain.lay1.biases = out[1]
    brain.lay2.weights = out[2]
    brain.lay2.biases = out[3]

    return brain

def test(brain):

    state = Board()

    while state.end == False:
        inputs = get_inputs(state)
        brain.forward(inputs)
        brain_out = brain.output
        foo = 0
        
        

        for i in range(4):
            prob = brain_out[0][i]
            if prob > foo:
                foo = prob
                move = i

        

        state.push(move)

        

    points = state.food_points + state.move_points

    return points

def display(brain):

    state = Board()
    root = tk.Tk()

    temp = 0
    
    while state.end == False:
        try:
            T.destroy()
        except:
            
            pass
        
        show = state.__str__()
        T = tk.Text(root)
        T.insert(tk.END,show)
        T.pack()

        inputs = get_inputs(state)
        brain.forward(inputs)
        brain_out = brain.output
        foo = 0

        for i in range(4):
            prob = brain_out[0][i]
            if prob > foo:
                foo = prob
                move = i    

        state.push(move)
        root.update()
        getter(root,'gif'+str(temp)+'.png')
        temp += 1
        
    points = state.food_points + state.move_points

    return points
    

def get_inputs(state):
    out = np.array([[0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0]],dtype = float)

    # d / l / u / r

    head = index(state.head)
    food = index(state.food)
    

    head_loc = (int(head // state.size),int(head % state.size))
    food_loc = (int(food // state.size),int(food % state.size))

    diff_x = head_loc[0] - food_loc[0]
    diff_y = head_loc[1] - food_loc[1]

    #print(head_loc)
    #print(food_loc)

    ouot = out[0]

    ouot[0] = head_loc[0]
    ouot[1] = head_loc[1]
    ouot[2] = state.size - ouot[0] - 1
    ouot[3] = state.size - ouot[1] - 1

    ouot[4] = (ouot[0] + ouot[1] - 1) / 2
    ouot[5] = (ouot[1] + ouot[2] - 1) / 2
    ouot[6] = (ouot[2] + ouot[3] - 1) / 2
    ouot[7] = (ouot[3] + ouot[0] - 1) / 2

    if head_loc[0] == food_loc[0]:

        if head_loc[1] > food_loc[1]:
            ouot[8] = head_loc[1] - food_loc[1]
        else:
            ouot[9] = food_loc[1] - head_loc[1]

    if head_loc[1] == food_loc[1]:
        if head_loc[0] > food_loc[0]:
            ouot[10] = head_loc[0] - food_loc[0]
        else:
            ouot[11] = food_loc[0] - head_loc[0]

    
    if diff_x == diff_y:
        if diff_x < 0:
            ouot[12] = abs(diff_x)
        else:
            ouot[13] = abs(diff_x)

    if diff_x == -diff_y:
        if diff_x < 0:
            ouot[14] = abs(diff_x)
        else:
            ouot[15] = abs(diff_x)


    for body in state.body_list:
        b = index(body)
        loc = (int(b // state.size),int(b % state.size))

        diff_x = head_loc[0] - loc[0]
        diff_y = head_loc[1] - loc[1]

        if loc[0] == head_loc[0]:
            if loc[1] > head_loc[1]:
                if ouot[16] == 0:
                    ouot[16] = loc[1] - head_loc[1]
                else:
                    ouot[16] = min(ouot[16],loc[1] - head_loc[1])

            else:
                if ouot[17] == 0:
                    ouot[17] = head_loc[1] - loc[1]
                else:
                    ouot[17] = min(ouot[17],head_loc[1] - loc[1])

        if loc[1] == head_loc[1]:
            if loc[0] > head_loc[0]:
                if ouot[18] == 0:
                    ouot[18] = loc[0] - head_loc[0]
                else:
                    ouot[18] = min(ouot[18],loc[0] - head_loc[0])

            else:
                if ouot[19] == 0:
                    ouot[19] = head_loc[0] - loc[0]
                else:
                    ouot[19] = min(ouot[19],head_loc[0] - loc[0])

        if diff_x == diff_y:
            if diff_x < 0:
                if ouot[20] == 0:
                    ouot[20] = abs(diff_x)
                else:
                    ouot[20] = min(ouot[20],abs(diff_x))
            else:
                if ouot[21] == 0:
                    ouot[21] = abs(diff_x)
                else:
                    ouot[21] = min(ouot[20],abs(diff_x))

        if diff_x == -diff_y:
            if diff_x < 0:
                if ouot[22] == 0:
                    ouot[22] = abs(diff_x)
                else:
                    ouot[22] = min(ouot[22],abs(diff_x))
            else:
                if ouot[23] == 0:
                    ouot[23] = abs(diff_x)
                else:
                    ouot[23] = min(ouot[23],abs(diff_x))
                

    for i in range(len(ouot)):
        if ouot[i] != 0:
            ouot[i] = (state.size - ouot[i] - 1) / (state.size - 2)

    return out

def plot_data(data):
    
    plt.clf()

    data = [d[0] for d in data]

    avg = [sum(d)/len(d) for d in data]
    maxx = [max(d) for d in data]
    std = [np.std(d) for d in data]

    low_var = [avg[i] - std[i] for i in range(len(avg))]
    high_var = [avg[i] + std[i] for i in range(len(avg))]

    #print(low_var)
    

    plt.plot(avg,'r',label = 'avg')
    plt.plot(maxx,label = 'max')
    plt.fill_between(range(len(avg)),low_var,high_var,label = 'std',color='grey')

    plt.legend()
    plt.ylabel('Fitness Score')
    plt.xlabel('Generation')
    
    
    plt.show(block=False)
    plt.pause(0.1)
    
def open_data(folder_name,gens):
    out = []

    for i in range(gens):

        brain_name = folder_name+'/Gen'+str(i)+'_Brain.npy'
        score_name = folder_name+'/Gen'+str(i)+'_Score.npy'

        brain = open_brain(brain_name)
        scores = np.load(score_name)

        out.append([scores,brain])

    return out
