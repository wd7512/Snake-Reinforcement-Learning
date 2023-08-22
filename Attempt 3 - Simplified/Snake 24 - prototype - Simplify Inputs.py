import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Brain: #generates random brain
    def __init__(self):
        # just inputs and outputs
        self.weights = np.random.randn(24,4)
        self.biases = np.zeros((1,4))

    def calc(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

    def mutate(self):
        self.weights = self.weights + 0.01 * np.random.randn(24,4)
        #self.biases = self.biases + 0.1 * np.random.randn(1,4)

#maybe need activation function


class Board:
    def __init__(self):
        self.size = 27
        self.half = int(np.floor(self.size/2))

        self.game_score = 0
        
        board = np.zeros((self.size,self.size),dtype = int)
        board[self.half-1,self.half] = 1
        board[self.half,self.half] = 2
        board[self.half+1,self.half] = 3
        self.game = board

    def add_food(self):
        food_pos = (np.random.randint(0,self.size),np.random.randint(0,self.size))


        while self.game[food_pos] != 0:
            food_pos = (np.random.randint(0,self.size),np.random.randint(0,self.size))

        self.game[food_pos] = -10

    def get_inputs(self):
        
        head_pos = np.argwhere(self.game == 1)[0]


        hx = head_pos[0]
        hy = head_pos[1]

        food_pos = np.where(self.game == -10)
        fx = food_pos[0]
        fy = food_pos[1]

        body_pos = np.where(self.game > 1)
        list_body_pos = list(zip(body_pos))
        

        wall_dist = []
        food_dist = [0,0,0,0,0,0,0,0]
        self_dist = [0,0,0,0,0,0,0,0]

        wall_dist.append(hx+1)
        wall_dist.append(21-hx)
        wall_dist.append(hy+1)
        wall_dist.append(21-hy)
        wall_dist.append(np.sqrt((hx+1)**2+(hy+1)**2))
        wall_dist.append(np.sqrt((hx+1)**2+(21-hy)**2))
        wall_dist.append(np.sqrt((21-hx)**2+(hy+1)**2))
        wall_dist.append(np.sqrt((21-hx)**2+(21-hy)**2))

        
        # output = [horizontal,vertical,non-leading,leading]
        
        #bottom left to top right must add to the same
        #leading diag must scale do diff in x = diff in y
        
        #horizontal first


        if hx == fx: #if horizontal plane
            if hy < fy:
                food_dist[0] = fy - hy
            else:
                food_dist[1] = hy - fy

        if hy == fy: #if vertical plane
            if hx < fx:
                food_dist[2] = fx - hx
            else:
                food_dist[3] = hx - fx


        if hx + hy == fx + fy: #is non-leading diag
            if hy < fy:
                food_dist[4] = (fy - hy)*np.sqrt(2)
            else:
                food_dist[5] = (hy - fy)*np.sqrt(2)

        if hx - fx == hy - fy: #leading diag
            if hy < fy:
                food_dist[6] = (fy - hy)*np.sqrt(2)
            else:
                food_dist[7] = (hy - fy)*np.sqrt(2)


        for pos in list_body_pos:

            #print(pos)
            #print(self.game)
            
            fx = pos[0][1] #swaped, might be wrong but seemed to be flipped
            fy = pos[0][0]
            

            if hx == fx: #if horizontal plane
                if hy < fy:
                    if fy - hy < self_dist[0] or self_dist[0] == 0:
                        self_dist[0] = fy - hy
                else:
                    if hy - fy < self_dist[1] or self_dist[1] == 0:
                        self_dist[1] = hy - fy

            if hy == fy: #if vertical plane
                if hx < fx:
                    if fx - hx < self_dist[2] or self_dist[2] == 0:
                        self_dist[2] = fx - hx
                else:
                    if hx - fx < self_dist[3] or self_dist[3] == 0:
                        self_dist[3] = hx - fx

            
            if hx + hy == fx + fy: #is non-leading diag
                if hy < fy:
                    if fy - hy < self_dist[4] or self_dist[4] == 0:
                        self_dist[4] = (fy - hy)*np.sqrt(2)
                else:
                    if hy - fy < self_dist[5] or self_dist[5] == 0:
                        self_dist[5] = (hy - fy)*np.sqrt(2)

            if hx - fx == hy - fy: #leading diag
                if hy < fy:
                    if fy - hy < self_dist[6] or self_dist[6] == 0:
                        self_dist[6] = (fy - hy)*np.sqrt(2)
                else:
                    if hy - fy < self_dist[7] or self_dist[7] == 0:
                        self_dist[7] = (hy - fy)*np.sqrt(2)


            
        

        
        max_dist = [self.size,self.size,self.size,self.size,
                    self.size*np.sqrt(2),self.size*np.sqrt(2),self.size*np.sqrt(2),self.size*np.sqrt(2)]

        tot_dist = np.asarray(max_dist + max_dist + max_dist)
                    
        
        # need to invert output

        self.inputs = (tot_dist - np.asarray(wall_dist + food_dist + self_dist,dtype = float)) / 100


    def run(self,brain):

        self.old_start = np.copy(self.game)

        self.past_states = []
        m = self.size * 3
        moves = m
        self.score = 0

        while moves > 0:

            moves = moves - 1
            self.score = self.score + 1

            self.get_inputs()
            brain.calc(self.inputs)

            output = brain.output
            ind_max = output.argmax()

            direc = (-1,0) #if 0, just saves an if statement
            if ind_max == 1:
                direc = (0,1)
            if ind_max == 2:
                direc = (1,0)
            if ind_max == 3:
                direc = (0,-1)

            self.past_states.append(np.copy(self.game))

            head_pos = np.where(self.game == 1)
            body_pos = np.where(self.game > 1)
            list_body_pos = list(zip(body_pos[0],body_pos[1]))
            snake_len = len(body_pos[0])

            new_head_pos = (head_pos[0][0]+direc[0],head_pos[1][0]+direc[1])


 

            on_itself = False

            for pos in list_body_pos:

                
                if new_head_pos == (pos[0],pos[1]):
     
                    on_itself = True


            if -1 == new_head_pos[0] or -1 == new_head_pos[1] or self.size == new_head_pos[0] or self.size == new_head_pos[1] or on_itself == True:
                moves = -1
            else:
                food_pos = np.where(self.game == -10)


                self.game[body_pos] = self.game[body_pos] + 1
                self.game[head_pos] = self.game[head_pos] + 1
                self.game[new_head_pos] = 1

                
                if new_head_pos == (food_pos[0],food_pos[1]):
                    
                    moves = moves + m
                    self.score = self.score + m
                    board = self.add_food()
                    self.game_score = self.game_score + 1
                    
                else:
                    self.game[np.where(self.game == (snake_len+2))] = 0

    def reset(self):
        self.game = np.copy(self.old_start)


def show_ani(frames): #frames may be the board states as a matrix
    fig = plt.figure()
    ims = []
    for frame in frames:
        im = plt.imshow(frame,animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    #ani.save('dynamic_images.gif')
    plt.show()

def simple_sim(Pop,Gens):

    repeat = 5

    top_percent = 0.1
    rand_percent = 0.1

    top_num = int(top_percent * Pop)
    rand_num = int(Pop * rand_percent)
    child_num = Pop - top_num - rand_num


    
    best_brains = []
    

    brains = [Brain() for i in range(Pop)]
    scores = np.zeros((Gens,Pop))
    

    for i in range(Gens):

        score = np.zeros((1,Pop))[0]
        game_score = np.zeros((1,Pop))[0]
        
        
        for j in range(repeat): #uses the same map for each repeat

            original = Board() #the start
            original.add_food()
            
            
            for k in range(Pop):

                brain = brains[k]
                original.run(brain)



                score[k] = original.score + score[k]
                game_score[k] = original.game_score = game_score[k]

                original.reset()

            
        score = score / (5 * original.size)

        
        # we have a list of brains
        # we have a list of their scores
        # dont use i,j,k
        # find top top_num brains
        # keep them and mutate
        # add randoms
        # save best brain video

        Var = np.copy(score) #copy to not mess up data

        best_game_score = max(game_score)
        best_score = max(Var)
        best_index = np.where(Var == best_score)[0][0]
        best_brain = brains[best_index]
        best_brains.append(best_brain)
        
        top_pos = [] # list of indexes of top_nums

        for x in range(top_num):
            highest = max(Var)
            highest_pos = np.where(Var == highest)[0][0]
            top_pos.append(highest_pos)

            Var[highest_pos] = 0

        top_scores = score[top_pos]

        top_brains = [brains[n] for n in top_pos]

        
        child_brains = []
        birthrate = int(np.floor(child_num / top_num))
        remainder = child_num - birthrate * top_num

        
        for brain in top_brains:
            for x in range(birthrate):
                Var = Brain() #new child
                Var.weights = brain.weights
                Var.biases = brain.biases

                Var.mutate()
                child_brains.append(Var)

        for x in range(remainder):
            Var = Brain() #new child
            Var.weights = best_brain.weights
            Var.biases = best_brain.biases

            Var.mutate()
            child_brains.append(Var)

        random_brains = [Brain() for x in range(rand_num)]

        

        brains = random_brains + top_brains + child_brains

        scores[i] = score

        
        print('\nGeneration: ',i)
        print('Population: ',len(brains))
        print('Best Game_Score: ',best_game_score/5)
        print('Best Score: ',best_score)
        print('Top Scores:  ',top_scores)
        
        
        
            

            

    return [scores,best_brains]

        
                
                
                
                

        
        
    


a = simple_sim(2000,100)

print('Complete')

game = Board()
game.add_food()
game.run(a[1][-1])
show_ani(game.past_states)
plt.matshow(a[0])
plt.colorbar()
plt.show()
