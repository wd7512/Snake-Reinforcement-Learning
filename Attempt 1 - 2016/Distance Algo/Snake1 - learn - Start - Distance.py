import turtle
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import ast


def emptymatrix(x,y): #makes empty matrix
    a=''
    for i in range(x):
        a=a+'0 '
    b=''
    for i in range(y-1):
        b=b+a+';'
    b=b+a

    return np.matrix(b)

def randombrain(population): #random networks

    brains=[]

    for i in range(population):
        net1=emptymatrix(14,16) #16 inputs
        net2=emptymatrix(4,14) #4 outputs
        for x in range(net1.shape[0]):
            for y in range(net1.shape[1]):
                net1[x,y]=random.randint(-100,100)
            
        for x in range(net2.shape[0]):
            for y in range(net2.shape[1]):
                net2[x,y]=random.randint(-100,100)
        
        brains.append([net1,net2])
    
    return brains

def savematrix(brain,generation): #saves in good form
    count=0
    for sub in brain:
        print(sub)
        f=open(str(generation)+'Layer'+str(count)+'.txt','w')
        save=[]
        for layer in sub:
            #print('L'+str(layer))
            save.append(str(layer))
        for i in range(len(save)):
            save[i]=save[i].split(' ')
       
        for s1 in save:
            #print(s1)
            s1=[x for x in s1
                if len(x)>0 and (str.isdigit(x[-1])==True
                                 or x[-2:]==']]')]
            #print(s1)

            #print(s1[0][0:2])
            
            if s1[0][0:2]=='[[':
                #print('bad')
                #print(s1[0][2:])
                s1[0]=s1[0][2:]

            #print(s1[-1][-2:])
            if s1[-1][-2:]==']]':
                #print('bad')
                #print(s1[0][2:])
                s1[-1]=s1[-1][:-2]
            
            f.write(str(s1)+'\n')
        f.close()
        count=count+1

def openmatrix(generation,layers): #number of generations already done and layers of network

    brain=[]
    for i in range(layers): #for each layer
        
        f=open(str(generation)+'Layer'+str(i)+'.txt','r')
        dat=f.readlines()


        data=[]

        for line in dat:
            data.append(ast.literal_eval(line))

       
        
        #print(data)

        matrix=emptymatrix(len(data[0]),len(data))

        #print(matrix)
            

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                #print(i)
                #print(j)
                matrix[i,j]=data[i][j]

        brain.append(matrix)
    return brain

def evolve(numberofbrains,brain):
    #print('population size'+str(numberofbrains))
    brains=[brain]
    change=random.randint(0,100)
    extras=0
    
    if change<6: #extra random mutations
        extras=round((numberofbrains-1)/(change+2))
        ranbrains=randombrain(extras)
        for ranbrain in ranbrains:
            brains.append(ranbrain)
        print(str(extras)+' have randomly mutated')

        
    for i in range(numberofbrains-1-extras):
        add1=emptymatrix(14,16)
        for i in range(16):
            for j in range(14):
                add1[i,j]=random.randint(-5,5)
        add2=emptymatrix(4,14)
        for i in range(14):
            for j in range(4):
                add2[i,j]=random.randint(-3,3)

        #print(brain[0])
        
        brains.append([brain[0]+add1,brain[1]+add2])

    for brain in brains:
        for subbrain in brain:
            for i in range(subbrain.shape[0]):
                for j in range(subbrain.shape[1]):
                    if subbrain[i,j]>100:
                        subbrain[i,j]=100
                    if subbrain[i,j]<-100:
                        subbrain[i,j]=-100
    return brains

def game(brain):
    turtle.clear()
    #20x20 "pixels"
    delay=0
    xdim=600 #must be multiple of 40
    ydim=600

    diag=math.sqrt(xdim**2+ydim**2)

    wn=turtle.Screen() #window
    wn.title('Snake')
    wn.bgcolor('Black') #background
    wn.setup(width=xdim, height=ydim) #screen size
    wn.tracer(0) #remove tracer

    def drawgrid():
        grid=turtle.Turtle()
        grid.speed(10)
        grid.color('green')
        grid.width(1)

        for i in range(int(xdim/20)):
            grid.penup()
            grid.goto(-xdim/2+i*20+10,ydim/2-10)
            grid.pendown()
            grid.goto(-xdim/2+i*20+10,-ydim/2+10)

        for i in range(int(ydim/20)):
            grid.penup()
            grid.goto(-xdim/2+10,ydim/2-10-i*20)
            grid.pendown()
            grid.goto(xdim/2-10,ydim/2-10-i*20)
        
    drawgrid()

    #head code
    head=turtle.Turtle()
    head.speed(0)
    head.shape('square')
    head.color('white')
    head.penup()
    head.goto(0,0)
    head.direction='stop'

    #food code
    food=turtle.Turtle()
    food.speed(0)
    food.shape('circle')
    food.color('red')
    food.penup()
    def movefood():
        xran=xdim/40
        yran=ydim/40
        food.goto(random.randint(-xran+1,xran-1)*20,random.randint(-yran+1,yran-1)*20) #go to random position

    segments=[]

    def move():
        if head.direction=='up':
            y=head.ycor()
            head.sety(y+20)

        if head.direction=='down':
            y=head.ycor()
            head.sety(y-20)

        if head.direction=='right':
            x=head.xcor()
            head.setx(x+20)

        if head.direction=='left':
            x=head.xcor()
            head.setx(x-20)

    def inputs():
        '''
        24 inputs
        8 directions * (distance to)[walls,itself,food]
        '''
        inputs=[]
        x=head.xcor()
        y=head.ycor()

        topwall=(ydim/2-20-y)/20
        botwall=ydim/20-2-topwall

        rightwall=(xdim/2-20-x)/20
        leftwall=xdim/20-2-rightwall

        toprightwall=round(math.sqrt(topwall**2+rightwall**2))
        botrightwall=round(math.sqrt(botwall**2+rightwall**2))
        botleftwall=round(math.sqrt(botwall**2+leftwall**2))
        topleftwall=round(math.sqrt(topwall**2+leftwall**2))

        fx=food.xcor()
        fy=food.ycor()

        topfood=0
        toprightfood=0
        rightfood=0
        botrightfood=0
        botfood=0
        botleftfood=0
        leftfood=0
        topleftfood=0

        
        if fx==x: #if on same x axis
            if fy>y:
                topfood=((fy-y)/20)
            if y>fy:
                botfood=((y-fy)/20)

        if fy==y: #if on same y axis
            if fx>x:
                rightfood=((fx-x)/20)
            if x>fx:
                leftfood=((x-fx)/20)

        if abs(x-fx)==abs(y-fy):
            if fx>x and fy>y: #topright
                topright=round(math.sqrt(2)*abs(x-fx))
            if fx>x and fy<y: #botright
                botright=round(math.sqrt(2)*abs(x-fx))
            if fx<x and fy<y: #botleft
                botleft=round(math.sqrt(2)*abs(x-fx))
            if fx>x and fy<y: #topleft
                topleft=round(math.sqrt(2)*abs(x-fx))
        
        

        return np.matrix(str(topwall)+' '+str(toprightwall)+' '+str(rightwall)+' '+str(botrightwall)+' '+str(botwall)+' '+str(botleftwall)+' '+str(leftwall)+' '+str(topleftwall)+' '+
                         str(topfood)+' '+str(toprightfood)+' '+str(rightfood)+' '+str(botrightfood)+' '+str(botfood)+' '+str(botleftfood)+' '+str(leftfood)+' '+str(topleftfood))
        
        #print(topwall)
        #print(botwall)

        #print(rightwall)
        #print(leftwall)
        
        

    def go_up():
        if head.direction!='down':
            print('up')
            head.direction='up'
        return
    def go_down():
        if head.direction!='up':
            print('down')
            head.direction='down'
        return
    def go_right():
        if head.direction!='left':
            print('right')
            head.direction='right'
        return
    def go_left():
        if head.direction!='right':
            print('left')
            head.direction='left'
        return

    def changemov(output):
        outlist=[]
        for i in range(4):
            outlist.append(output[0,i])

        high=outlist.index(max(outlist))

        if high==0:
            go_up()
            
        if high==1:
            go_down()
            
        if high==2:
            go_right()
            
        if high==3:
            go_left()

        return

    def end(tim):
        turtle.clearscreen()
        return tims

    locked=time.time()
    tims=0
    while True: #update loop
        
        wn.update()

        #print(head.direction)

        if abs(head.xcor())>xdim/2-10 or abs(head.ycor())>ydim/2-10: #if hits boundries
            time.sleep(0.25)

            return end(tims)
            
            head.goto(0,0) #restart
            head.direction='stop' #stop moving

            for segment in segments:
                segment.color('black') #blend old segments into background
                segment.goto(xdim,ydim)
            segments=[] #get empty segments
            
            

        if head.distance(food)<20: #if on food
            movefood()

            tims=tims+0 #food reward
            
            new_segment=turtle.Turtle()
            new_segment.speed(0)
            new_segment.shape('square')
            new_segment.color('grey')
            new_segment.penup()
            
            segments.append(new_segment)
        #move end segments

        for i in range(len(segments)-1,0,-1): #0,-1 reverses order of i
            x=segments[i-1].xcor()
            y=segments[i-1].ycor()

            segments[i].goto(x,y) #moves to one in front

        if len(segments)>0:
            x=head.xcor()
            y=head.ycor()
            segments[0].goto(x,y)

        inp=inputs()

        hiddenlayer=inp*brain[0]

        print(hiddenlayer)

        output=hiddenlayer*brain[1]

        print(output)

        changemov(output)

        move()

        for segment in segments:
            if head.distance(segment)<20: #if hits body
                time.sleep(1)

                #end()
                
                head.goto(0,0) #restart
                head.direction='stop' #stop moving

                for segment in segments:
                    segment.color('black') #blend old segments into background
                    segment.goto(xdim,ydim)
                segments=[] #get empty segments
                
        tims=tims+round(diag-head.distance(food)) #living reward
                                
        time.sleep(delay)


        if time.time()-locked>10:
            return end(tims)

    wn.mainloop() #keepwindow open

def play(pop,gens,brain):

    #brains=evolve(pop,brain)
    brains=randombrain(pop)

    for i in range(gens):
        print('\nGeneration'+str(i+1)+'\n')
        scores=[]
        for brain in brains:
            scores.append(game(brain))

        bestnum=scores.index(max(scores))
        bestbrain=brains[bestnum]

        brains=evolve(pop,bestbrain)

    return bestbrain
'''
gen=int(input('Current gen:'))
layers=2

brain=openmatrix(gen,layers)

pop=int(input('population:'))
newgens=int(input('extra gens:'))

bestbrain=play(pop,newgens,brain)

savematrix(bestbrain,gen+newgens)
'''
savematrix(play(1000,1,0),0)
