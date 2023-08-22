import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
from copy import deepcopy
from datetime import datetime

'''
ok lets keep things as simple as possible

board = 25x25 with snake size 1

what if we only had 16 inputs
4 for food - horizontal, vertical, 2 diags
and 8 for walls / body - go away

neural network design
12 inputs
24 layer
4 outputs

on output we take the max of the 4 long list
[left,up,right,down]
'''

def new_board():
    bo = np.zeros((25,25),dtype=int)
    bo[11,12] = 1
    bo[12,12] = 2
    bo[13,12] = 3

    bo = add_food(bo)

    return bo

def add_food(bo):
    zeros = np.where(bo == 0)
    zeros = list(zip(zeros[0],zeros[1]))
    
    loc = random.choice(zeros)
    bo[loc] = -1

    return bo

def get_inputs_12(bo):
    head = np.where(bo == 1)
    head_pos = (head[0][0],head[1][0])

    out = np.zeros(12,dtype=float)

    out[0] = head_pos[0] #distance from top 
    out[1] = head_pos[1] #distance from left
    out[2] = 24 - head_pos[0] #distance from bottom
    out[3] = 24 - head_pos[1] #distance from right
    out[4] = min(out[0],out[1]) * np.sqrt(2) #distance from top-left
    out[5] = min(out[0],out[3]) * np.sqrt(2) #distance from top-right
    out[6] = min(out[2],out[3]) * np.sqrt(2) #distance from bot-right
    out[7] = min(out[2],out[1]) * np.sqrt(2) #distance from bot-left

    food = np.where(bo == -1)
    food_pos = (food[0][0],food[1][0])

    diff_y = food_pos[0]-head_pos[0]
    diff_x = food_pos[1]-head_pos[0]

    if food_pos[0] == head_pos[0]: #if in same row
        out[8] = food_pos[1] - head_pos[1]
    elif food_pos[1] == head_pos[1]: #if in same col
        out[9] = food_pos[0] - head_pos[0]

    
    elif abs(diff_y) == abs(diff_x): #if on diag
        if diff_y == diff_x:
            out[10] = diff_y * abs(diff_x)
        else:
            out[11] = diff_y * abs(diff_x)

    body = np.where(bo > 0)
    body_pos = list(zip(body[0],body[1]))
        
    

    return out

def get_inputs_16(bo):
    head = np.where(bo == 1)
    head_pos = (head[0][0],head[1][0])

    out = np.zeros(16,dtype=float)

    out[0] = head_pos[0] #distance from top 
    out[1] = head_pos[1] #distance from left
    out[2] = 24 - head_pos[0] #distance from bottom
    out[3] = 24 - head_pos[1] #distance from right
    out[4] = min(out[0],out[1]) * np.sqrt(2) #distance from top-left
    out[5] = min(out[0],out[3]) * np.sqrt(2) #distance from top-right
    out[6] = min(out[2],out[3]) * np.sqrt(2) #distance from bot-right
    out[7] = min(out[2],out[1]) * np.sqrt(2) #distance from bot-left

    food = np.where(bo == -1)
    food_pos = (food[0][0],food[1][0])

    diff_y = food_pos[0]-head_pos[0]
    diff_x = food_pos[1]-head_pos[0]

    if food_pos[0] == head_pos[0]: #if in same row
        diff = food_pos[1] - head_pos[1]
        if diff > 0:
            out[8] = diff
        else:
            out[9] = abs(diff)
    elif food_pos[1] == head_pos[1]: #if in same col
        diff = food_pos[0] - head_pos[0]
        if diff > 0:
            out[10] = diff
        else:
            out[11] = abs(diff)

    
    if diff_y == diff_x:
        if diff_y > 0:
            out[12] = diff_y * np.sqrt(2)
        else:
            out[13] = abs(diff_y) * np.sqrt(2)
    elif abs(diff_y) == abs(diff_x):
        if diff_y > 0:
            out[14] = diff_y * np.sqrt(2)
        else:
            out[15] = abs(diff_y) * np.sqrt(2)

    body = np.where(bo > 0)
    body_pos = list(zip(body[0],body[1]))
        
    

    return out

class Layer_Dense:
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
        self.lay1 = Layer_Dense(16,24)
        self.act1 = Activation_ReLU()
        self.lay2 = Layer_Dense(24,4)
        self.act2 = Activation_Softmax()
    def forward(self, inputs):
       
        self.lay1.forward(inputs)
        self.act1.forward(self.lay1.output)
        
        self.lay2.forward(self.act1.output)
        self.act2.forward(self.lay2.output)

        self.output = self.act2.output

def run(net):
    states = [new_board()]

    points = 0
    moves = 100
    while moves > 0:
        bo = states[-1]
        zeros = np.where(bo == 0)
        zeros = list(zip(zeros[0],zeros[1]))

        inputs = get_inputs_16(bo)
        net.forward(inputs)

        move = net.output.argmax()
        
        if move == 0:
            direc = (0,-1)
        elif move == 1:
            direc = (-1,0)
        elif move == 2:
            direc = (0,1)
        elif move == 3:
            direc = (1,0)
        else:
            print('illegal what!?')

        new_bo = np.copy(bo)

        head_pos = np.where(bo == 1)
        body_pos = np.where(bo > 1)
        list_body_pos = list(zip(body_pos[0],body_pos[1]))

        new_head_pos = (head_pos[0]+direc[0],head_pos[1]+direc[1])

        for pos in list_body_pos:
            if new_head_pos == (pos[0],pos[1]):
                moves = -1
                break

        if -1 == new_head_pos[0] or -1 == new_head_pos[1] or 25 == new_head_pos[0] or 25 == new_head_pos[1]:
            moves = -1
            break
        else:
            food_pos = np.where(bo == -1)

            new_bo[body_pos] = new_bo[body_pos] + 1
            new_bo[head_pos] = new_bo[head_pos] + 1
            new_bo[new_head_pos] = 1

            if new_head_pos == (food_pos[0],food_pos[1]):
                new_bo = add_food(new_bo)
                moves = moves + 100
                points = points + 1000

            else:
                new_bo[np.where(new_bo == (len(list_body_pos)+2))] = 0


        states.append(new_bo)
        moves = moves - 1
        points = points + 1

    return states, points

def show_ani(frames): #frames may be the board states as a matrix
    fig = plt.figure()
    ims = []
    for frame in frames:
        im = plt.imshow(frame,animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    #ani.save('dynamic_images.gif')
    plt.show()

def mutate(brain, n):
    out = [brain]

    for i in range(n-1):
        new_brain = deepcopy(brain)

        new_brain.lay1.weights += 0.01 * np.random.randn(16, 24)
        #new_brain.lay1.biases += 0.01 * np.random.randn(1,24)
        new_brain.lay2.weights += 0.01 * np.random.randn(24, 4)
        #new_brain.lay2.biases += 0.01 * np.random.randn(1,4)
        
        out.append(new_brain)

    return out

def multi_mutate(brains, children):
    out = brains

    for i in range(len(brains)):
        brain = brains[i]
        for i in range(children):
            new_brain = deepcopy(brain)

            new_brain.lay1.weights += np.random.randn(16, 24) * np.random.randn(16, 24)
            #new_brain.lay1.biases += 0.01 * np.random.randn(1,24)
            new_brain.lay2.weights += np.random.randn(24, 4) * np.random.randn(24, 4)
            #new_brain.lay2.biases += 0.01 * np.random.randn(1,4)
        
            out.append(new_brain)

    return out

def learn(networks,gens,best=None):

    trials = 10

    top_brains = int(networks/10)

    if best == None:
        trial_brains = [Brain() for i in range(networks)]
    else:
        trial_brains = multi_mutate([best], networks-1)
        
    output = []

    for i in range(gens):
        games = []
        scores = []
        for brain in trial_brains:
            
            avg_score = 0
            best_score = 0
            for j in range(trials):
                frames,score = run(brain)

                avg_score += score/trials
                
                if score > best_score:
                    best_score = score
                    best_game = frames

            games.append(best_game)
            scores.append(best_score)

        print('Gen :',i)
        print('Best Score :',max(scores))
        print('Avg Score :',sum(scores) / len(scores))

        max_index = scores.index(max(scores))
        #show_ani(games[max_index])

        best_brain = trial_brains[max_index]

        output.append([games[max_index],scores,best_brain])

        scores_copy = scores.copy()
        multi_best_brains = []
        for x in range(top_brains):
            multi_best_brains.append(trial_brains[scores_copy.index(max(scores_copy))])

            scores_copy.remove(max(scores_copy))


        trial_brains = multi_mutate(multi_best_brains, 9)

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


def prog_learn(networks,gens,best=None):

    save_rate = 10

    trials = 10

    top_brains = int(networks/10)

    if best == None:
        trial_brains = [Brain() for i in range(networks)]
    else:
        trial_brains = multi_mutate([best], networks-1)
        
    output = []

    for i in range(gens):
        games = []
        scores = []
        for brain in trial_brains:
            
            avg_score = 0
            best_score = 0
            for j in range(trials):
                frames,score = run(brain)

                avg_score += score/trials
                
                if score > best_score:
                    best_score = score
                    best_game = frames

            games.append(best_game)
            scores.append(best_score)

        print('Gen :',i)
        print('Best Score :',max(scores))
        print('Avg Score :',sum(scores) / len(scores))

        max_index = scores.index(max(scores))
        #show_ani(games[max_index])

        best_brain = trial_brains[max_index]

        output.append([games[max_index],scores,best_brain])

        scores_copy = scores.copy()
        multi_best_brains = []
        for x in range(top_brains):
            foo = trial_brains[scores_copy.index(max(scores_copy))]
            multi_best_brains.append(foo)
            if (i+1) % save_rate == 0:
                save_brain(foo,'brains/A-Gen-'+str(i)+'-Top-'+str(x))
            

            scores_copy.remove(max(scores_copy))


        trial_brains = multi_mutate(multi_best_brains, 1)

    return output

        

