import math
import random

def index(b):
    return math.log(b)/math.log(2)

def bini(index):
    return 2 ** index


class Board():
    def __init__(self):
        self.size = 15
        self.full_size = self.size ** 2

        self.walls = 0

        for i in range(self.size):
            self.walls += bini(i)
            self.walls += bini(self.size*(self.size-1)+i)

        for i in range(self.size-2):
            self.walls += bini(self.size + self.size * i)
            self.walls += bini(self.size *2 -1 + self.size * i)

        self.food = 0
        self.head = bini(round(self.full_size / 2)) << self.size
        self.body_list = [(self.head >> self.size*2) , (self.head >> self.size)]

        self.food_points = 0
        self.move_points = 0
        self.energy = 100

        self.end = False

        self.update()

        self.place_food()

    def __str__(self):
        con = '{:'+str(self.full_size)+'b}'
        walls = con.format(self.walls)
        head = con.format(self.head)
        food = con.format(self.food)
        body = con.format(self.body)

        rows = []
        r = ''
        for i in range(self.full_size):
            r += ' '
            if walls[i] == '1':
                r = r + 'X'
            elif head[i] == '1':
                r = r + 'H'
            elif food[i] == '1':
                r = r + 'F'
            elif body[i] == '1':
                r = r + 'B'
            
            else:
                r = r + ' '

            if (i+1) % self.size == 0:
                rows.append(r[::-1])
                r = ''

        out = '\n'.join(rows)

        return out
        
    def update(self):
        self.body = sum(self.body_list)

        self.all = self.walls | self.food | self.body | self.head

        if self.head & self.body != 0:
            self.end = True
            #print('GAME OVER')

        if self.head & self.walls != 0:
            self.end = True
            #print('GAME OVER')

        if self.energy < self.move_points:
            self.end = True

    def place_food(self):
        choices = []
        for i in range(self.full_size):
            if bini(i) & self.all == 0:
                choices.append(i)

        loc = random.choice(choices)
        self.food = bini(loc)

    def print(self,var):
        con = '{:'+str(self.full_size)+'b}'
        return con.format(var)

    def push(self,move):
        # 0 - Left
        # 1 - UP
        # 2 - Right
        # 3 - DOWN

        old_head = self.head

        if move == 0:
            self.head = self.head >> 1
        elif move == 1:
            self.head = self.head << self.size
        elif move == 2:
            self.head = self.head << 1
        elif move == 3:
            self.head = self.head >> self.size
        else:
            self.end = True
            print('wtf is this input')

        if self.head & self.food == 0:#if not on a food
            self.body_list.remove(self.body_list[0])
            self.body_list.append(old_head)

        else:
            self.food_points += 1000
            self.energy += 100
            self.body_list.append(old_head)
            self.place_food()

        self.update()
        self.move_points += 1
    
