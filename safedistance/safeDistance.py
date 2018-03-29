def safeDistance(currSpeed, relSpeed, currDistance):
    # All speeds are in m/s
    #assuming that the vehicle should get in the safe distance range within 10 seconds
    tReact= 1.5
    length=4.80
    a=3.4
    b=4.5
    t1=0
    t2=t1+tReact
    stoppingDistance= 0.5*(((currSpeed*currSpeed)/b)- ((currSpeed+relSpeed)*(currSpeed+relSpeed)/a))+(currSpeed*tReact)
    safeDistance= stoppingDistance+length
    relative= currDistance-safeDistance
    safeVel= (currSpeed+ relSpeed)+(relative/15)
    return (safeDistance, safeVel)

'''def safeVelocity(currDis
    vRelatice= velA-velB
    safeD=safeDistance(velB)
    currentD= currentDistance(GPSA, GPSB)
    relative= safeD-currentD
    if currentD<safeD:
        print("Safe velocity: ",velA-(relative/15))
    else:
        print("Maximum Safe velocity: ", velA+(relative/15))'''

Speed=[5,-5,10,-10,2,-2,4,-4,0,1]
Distance= [1,2,3,4,5,6,7,8,9,10]
for i in range(10):
    currSpeed=18
    relSpeed=Speed[i]
    currDistance=Distance[i]
    print(safeDistance(currSpeed, relSpeed, currDistance))
    
