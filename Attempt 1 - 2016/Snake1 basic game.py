import turtle
import time
import random

def game():
    #20x20 "pixels"
    delay=0.03
    xdim=600 #must be multiple of 40
    ydim=600

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

    def go_up():
        head.direction='up'
    def go_down():
        head.direction='down'
    def go_right():
        head.direction='right'
    def go_left():
        head.direction='left'
    wn.listen()

    wn.onkeypress(go_up,'w')
    wn.onkeypress(go_down,'s')
    wn.onkeypress(go_right,'d')
    wn.onkeypress(go_left,'a')
        

    while True: #update loop
        wn.update()

        #print(head.direction)

        if abs(head.xcor())>xdim/2-10 or abs(head.ycor())>ydim/2-10: #if hits boundries
            time.sleep(1)
            head.goto(0,0) #restart
            head.direction='stop' #stop moving

            for segment in segments:
                segment.color('black') #blend old segments into background
                segment.goto(xdim,ydim)
            segments=[] #get empty segments
            
            

        if head.distance(food)<20: #if on food
            movefood()

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

        move()

        for segment in segments:
            if head.distance(segment)<20: #if hits body
                time.sleep(1)
                head.goto(0,0) #restart
                head.direction='stop' #stop moving

                for segment in segments:
                    segment.color('black') #blend old segments into background
                    segment.goto(xdim,ydim)
                segments=[] #get empty segments
        time.sleep(delay)

    wn.mainloop() #keepwindow open
game()
