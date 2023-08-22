import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def new_board(size): #creates empty game board
    half = math.floor(size/2)
    board = np.zeros((size,size),dtype = int)
    board[half-1,half] = 1
    board[half,half] = 2
    board[half+1,half] = 3

    
    food_space = [random.randint(0,size-1),random.randint(0,size-1)]
    while food_space == [half-1,half]:
        food_space = [random.randint(0,size-1),random.randint(0,size-1)]
    board[food_space[0],food_space[1]] = -10
    
    
    return board
    
def add_food(board,size): # had potential to be faster
    flat = (board+0).reshape([1,size*size])
    num = np.random.choice(np.where(flat==0)[1])
    flat[0,num] = -10
    mat = flat.reshape([size,size])
    return mat

def show(frames): #frames may be the board states as a matrix
    fig = plt.figure()
    ims = []
    for frame in frames:
        im = plt.imshow(frame,animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    #ani.save('dynamic_images.gif')
    plt.show()

def test_food(no_frames,size):
    boards = [new_board(size)]
    for i in range(no_frames-1):
        new = add_food(boards[-1],size)
        boards.append(new)
    show(boards)
    return boards
    


def get_inputs(board,size):
    # 24 inputs 8 dir * distance to [wall,food,itself]
    half = math.floor(size/2) 
    flat = (board+0).reshape([1,size*size])
    
    pos = np.where(board == 1)
    #print(pos)
    head_pos = (pos[0][0],pos[1][0])


    v_slice = board[:,head_pos[1]]
    h_slice = board[head_pos[0]]

    dif_x = half - head_pos[0]
    dif_y = half - head_pos[1]

    K = (dif_x-dif_y,-dif_y-dif_x)
    #print(K)
    d1_slice = np.diag(board,k=K[0]) 
    d2_slice = np.diag(np.fliplr(board+0),k=-K[1])
    '''
    print(v_slice)
    print(h_slice)
    print(d1_slice)
    print(d2_slice)
    print(board)
    '''
    vA = np.flip(v_slice[:head_pos[0]])
    vB = v_slice[head_pos[0]+1:]
    hA = np.flip(h_slice[:head_pos[1]])
    hB = h_slice[head_pos[1]+1:]

    
    
    d1A = np.flip(d1_slice[:np.where(d1_slice==1)[0][0]])
    d1B = d1_slice[np.where(d1_slice==1)[0][0]+1:] 
    d2A = np.flip(d2_slice[:np.where(d2_slice==1)[0][0]])
    d2B = d2_slice[np.where(d2_slice==1)[0][0]+1:]

    slices = [vA,d2A,hB,d1B,vB,d2B,hA,d1A]
    '''
    print(vA) 
    print(d2A)
    print(hB) 
    print(d1B) 
    print(vB)
    print(d2B)
    print(hA)
    print(d1A)
    '''

    #output = [wall dist,food dist,iteself dist]
    output = (np.zeros((1,24)))
    #print(output)
    output[0][0] = pos[0][0] + 1
    output[0][1] = math.sqrt(2) * (pos[0][0] + 1 + pos[1][0] + 1)
    output[0][2] = pos[1][0] + 1
    output[0][3] = math.sqrt(2) * (size - (pos[0][0]) + pos[1][0] + 1)
    output[0][4] = size - (pos[0][0])
    output[0][5] = math.sqrt(2) * (size - (pos[0][0]) + size - (pos[1][0]))
    output[0][6] = size - (pos[1][0])
    output[0][7] = math.sqrt(2) * (size - (pos[1][0]) + pos[0][0] + 1)

    #print(output)

    inf = 10**6

    slices_self = [np.where(var > 1)[0][0]+1 if any(x>1 for x in var) else inf for var in slices]
    #print(slices_self)
    slices_food = [np.where(var==-10)[0][0]+1 if -10 in var else inf for var in slices]
    #print(slices_food)

    for i in range(8):
        if i%2 == 1:
            output[0][i+8] = slices_self[i] * math.sqrt(2)
        else:
            output[0][i+8] = slices_self[i]
    for i in range(8):
        if i%2 == 1:
            output[0][i+16] = slices_food[i] * math.sqrt(2)
        else:
            output[0][i+16] = slices_food[i]

    #print(output)

    return output
        
def function(x):
    return 1/(1+np.exp(-x))
    
def random_brain():
    # 24 - 4

    lay1 = np.random.uniform(-1,1,(24,4))
    add1 = np.random.uniform(-1,1,(1,4))


    return [lay1,add1]

def run_brain(brain,inputs):
    output = function(np.matmul(inputs,brain[0])) + brain[1]

    return output #[top, right, bot, left]

def run_game(brain):

    states = []
    
    size = 41
    board = new_board(size)

    moves = 100
    score = 0

    while moves > 0:
        moves = moves - 1
        score = score + 1
        
        inputs = get_inputs(board,size)
        outputs = run_brain(brain,inputs)
        ind_max = outputs.argmax()
        
        direc = (-1,0)
        if ind_max == 1:
            direc = (0,1)
        if ind_max == 2:
            direc = (1,0)
        if ind_max == 3:
            direc = (0,-1)

        #print(board)
        states.append(board+0)
        #print(direc)
        

        head_pos = np.where(board == 1)
        body_pos = np.where(board > 1)
        #print(body_pos)
        snake_len = len(body_pos[0])
        
        
        new_head_pos = (head_pos[0]+direc[0],head_pos[1]+direc[1])

        
        on_itself = False
        body_pos_list = []
        for i in range(snake_len):

            body_pos_list.append((body_pos[0][i],body_pos[1][i]))

            if np.all(new_head_pos == body_pos_list[-1]):
                on_itself = True
        #print(new_head_pos)
        if -1 in new_head_pos[0] or -1 in new_head_pos[1] or size in new_head_pos[0] or size in new_head_pos[1] or on_itself == True:
            moves = -1
        else:
            food_pos = np.where(board == -10)

            board[body_pos] = board[body_pos] + 1
            board[head_pos] = board[head_pos] + 1
            board[new_head_pos] = 1

            
            if new_head_pos == (food_pos[0],food_pos[1]):
                
                moves = moves + 100
                score = score + 10
                board = add_food(board,size)
                
            else:
                board[np.where(board == (snake_len+2))] = 0
    
    #print(score)
    return [score,states,brain]

def mutate(brain):
    factor = 10**-4
    new_brain = [sub + (np.random.uniform(-1,1,(np.shape(sub)[0],np.shape(sub)[1])))*factor for sub in brain]
    return new_brain
    
def basic_sim(pop,gens):

    best_of_gen = []
    
    pop_brains = [random_brain() for i in range(pop)]
    pop_games = [run_game(brain) for brain in pop_brains]
    pop_games = sorted(pop_games, reverse = True, key=lambda x: x[0])

    top_percent = 0.05    
    rand_percent = 0.4

    top_nums = int(top_percent * pop)
    randoms = int(rand_percent * pop)
    child_percent = 1 - top_percent - rand_percent
    child_num = round(child_percent / top_percent)
    '''
    print(child_num)
    print(randoms)
    print(top_nums)
    '''
    best = pop_games[:top_nums]
    best_of_gen.append(best[0])

    #show(best_of_gen[-1][1])

    for i in range(gens):
        print('Gen = '+str(i))
        
        pop_brains = [b[2] for b in best] + [random_brain() for i in range(randoms)]
        
        
        for person in best:
            brain = person[2]
            children = [mutate(brain) for i in range(child_num)]
            pop_brains = pop_brains + children

        print('population = '+str(len(pop_brains)))


        pop_games = [run_game(brain) for brain in pop_brains]
        pop_games = sorted(pop_games, reverse = True, key=lambda x: x[0])
        #print(pop_games[0])

        best = pop_games[:top_nums]

        best_of_gen.append(best[0])

        print('Best Score = '+str(best[0][0]))

        #show(best_of_gen[-1][1])


    return best_of_gen

def averaged_sim(pop,gens): #runs each brain x times and takes an average

    def avg_run(brain):
        
        runs = 3
        
        all_runs = [run_game(brain) for i in range(5)]
        all_runs = sorted(all_runs, reverse = True, key=lambda x: x[0])
        best = all_runs[0]

        total_score = 0
        for run in all_runs:
            total_score = total_score + run[0]

        total_score = int(total_score/runs)

        return [total_score,best[1],brain]


    best_of_gen = []

    pop_brains = [random_brain() for i in range(pop)]
    pop_games = [avg_run(brain) for brain in pop_brains]
    pop_games = sorted(pop_games, reverse = True, key=lambda x: x[0])

    top_percent = 0.1    
    rand_percent = 0.4

    top_nums = int(top_percent * pop)
    randoms = int(rand_percent * pop)
    child_percent = 1 - top_percent - rand_percent
    child_num = round(child_percent / top_percent)

    best = pop_games[:top_nums]
    best_of_gen.append(best[0])

    for i in range(gens):
        print('Gen = '+str(i))
        
        pop_brains = [b[2] for b in best] + [random_brain() for i in range(randoms)]
        
        
        for person in best:
            brain = person[2]
            children = [mutate(brain) for i in range(child_num)]
            pop_brains = pop_brains + children

        print('population = '+str(len(pop_brains)))


        pop_games = [avg_run(brain) for brain in pop_brains]
        pop_games = sorted(pop_games, reverse = True, key=lambda x: x[0])
        #print(pop_games[0])

        best = pop_games[:top_nums]

        best_of_gen.append(best[0])

        print('Best Score = '+str(best[0][0]))

        

        #show(best_of_gen[-1][1])


    return best_of_gen
    

def save_matrix(mat,name):
    f = open(name+'.csv','w')
    size = np.shape(mat)
    #print(mat)
    for i in range(size[0]):
        line = str(mat[i])
        
        line = line.replace(']','')
        line = line.replace('[','')
        line = line.replace('    ',',')
        line = line.replace('   ',',')
        line = line.replace('  ',',')
        line = line.replace(' ',',')
        line = line.replace('\n','')
        if line[0] == ',':
            line = line[1:]
        #print(line)
        
        f.write(line+'\n')
    f.close()

def open_matrix(name):
    if '.csv' in name:
        pass
    else:
        name = name+'.csv'
    f = open(name,'r')
    data = f.readlines()
    new_data = []
    for d in data:
        new_data.append((d.replace('\n','')).split(','))
    

    
    for d in new_data:
        while '' in d:
            d.remove('')

    #print(new_data)

    size = (len(new_data),len(new_data[0]))
    print(size)
    
    mat = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            try:
                flt = new_data[i][j]
            except IndexError:
                print(name)

            mat[i][j] = float(flt)

    return mat
    
def save_brain(brain,name):
    save_matrix(brain[0],name+'_lay1')
    save_matrix(brain[1],name+'_add1')
    save_matrix(brain[2],name+'_lay2')
    save_matrix(brain[3],name+'_add2')

def open_brain(name):
    lay1 = open_matrix(name+'_lay1')
    add1 = open_matrix(name+'_add1')
    lay2 = open_matrix(name+'_lay2')
    add2 = open_matrix(name+'_add2')

    return [lay1,add1,lay2,add2]

def save_frames(frames,name):
    #print(len(frames))
    for i in range(len(frames)):
        frame = frames[i]
        save_name = name+'_frame_'+str(i)
        save_matrix(frame,save_name)

def open_frames(name):
    frames = []
    all_files = os.listdir()
    find_name = name + '_frame_'
    frame_files = []
    for name in all_files:
        if find_name in name:
            frame_files.append(name)

    sorted_frame_files = []
    for i in range(len(frame_files)):


        for j in range(len(frame_files)):
            file = frame_files[j]

            to_find = '_'+str(i)+'.csv'
            
            if to_find in file:
                #print(file)
                found = True
                sort_file = file

                sorted_frame_files.append(sort_file)

    #print(sorted_frame_files)

    
    for file in sorted_frame_files:
        frames.append(open_matrix(file))

    return frames
                
'''
sim = basic_sim(100,10)
final = sim[-1] #[score,frames,brain]
final_frames = final[1]
final_brain = final[2]

b = random_brain()
save_brain(b,'new')
a = open_brain('new')

show(run_game(b))
show(run_game(a))
'''


sim = averaged_sim(100,100)
a = sim[-1]
save_frames(a[1],'avg')
save_brain(a[2],'avg')





