import math
from math import sqrt
import sys
import numpy as np
import utilities as util
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation

def headInSameDirection(x1,x2,y1,y2):
    return (x1<0) == (x2<0) and (y1<0) == (y2<0)

def getVelocity(current, prev):
    return (30 * (current[0] - prev[0]), 30 * (current[1] - prev[1]))



def particleFilterPrediction(inputData,allData):
    
    output = []
    #Indice for point instance in input data that has the best match to that from training data
    bestMatchId = -1
    bestMatch = -1
    matchDiff = 5000
    #number of points we are comparing
    N = 20

    for f in range(0,len(allData)):
        #instance of trainingData
        current = allData[f]
        inputPoints = inputData[len(inputData)-N:]
        #Loop the training data for best match
        for i in range(N-1,len(current)-60):

            training = current[i-N+1:i+1]
            # change of point differences(velocity) for last two instances of training -data
            xTrainV=getVelocity(training[N-1],training[N-2])[0]
            yTrainV=getVelocity(training[N-1],training[N-2])[1]
            
            
            # change of point differences(velocity) for last two instances of input -data
            xInputV=getVelocity(inputPoints[N-1],inputPoints[N-2])[0]
            yInputV=getVelocity(inputPoints[N-1],inputPoints[N-2])[1]

            
            # make sure heading are same
            if (headInSameDirection(xTrainV,xInputV,yTrainV,yInputV)):
                # totalDiff is the total change in distance between the input and training N points
                totalDiff = 0
                for n in range(N):
                    totalDiff += pointsDistance(inputPoints[n], training[n])
                # update for min match score
                if totalDiff < matchDiff:
                    bestMatchId = f
                    bestMatch = i
                    matchDiff = totalDiff

    #calculate the differences of the last 60 frames after the match point

    xdiff = []
    ydiff = []
    
    #calculate the point diffs of those points that are 60 frames after the best match and then fill xdiff and ydiff to the output
    for i in range(bestMatch+1, bestMatch+61):
        prev = allData[bestMatchId][i-1]
        current = allData[bestMatchId][i]
        xdiff.append(current[0] - prev[0])
        ydiff.append(current[1] - prev[1])
    prev = inputData[len(inputData)-1]
    for i in range(len(xdiff)):
        output.append([prev[0] + xdiff[i], prev[1] + ydiff[i]])
        prev = output[len(output) - 1]
   
    return output




def pointsDistance(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


#function that parses txt into array
def parseTxt(filename):

	f = open(filename,"r")
        arr=[]
        for l in f:
            measurement=l.replace("\r\n", "").split(',')
            #######need int here#####################
	    measurement=[int(measurement[0]),int(measurement[1])]
            arr.append(measurement)
        return arr


def consolidateAllData():
    allData=[]
    # Below we added everything(training and 10 inputs) into a list for later processcing/matching
    # Add in training_data
    trainingData=parseTxt("inputs/" + "training_data.txt")
    allData.append(trainingData[:len(trainingData)-60])
    # Add ten input data
    for i in range (1, 11):
        fileName = "test" + ("%02d" % (i,)) + ".txt"
        inputFile=parseTxt("inputs/" + fileName)
        allData.append(inputFile[:len(inputFile)-60])
    return allData

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
	#parse the input txt,eg:text01.txt
        givenData=parseTxt(filename)

        
        
         
        #use the last 60 frames of gieven input, eg test01.txt as actual
        actual=givenData[len(givenData)-60:]
        
        # feed the program input data with last 60 frames chopped off
        allData=consolidateAllData()
        output=particleFilterPrediction(givenData[:len(givenData)-60],allData)
        
        #####need to write output to txt#########

        
        #use utility to test our error and make plots
        errors=[]
        for i in range(0,len(output)):
            #errors.append(util.error([actual[i]],[output[i]]))
            ok=((output[i][0]-actual[i][0])**2) + ((output[i][1]-actual[i][1])**2)
            errors.append(ok)

        util.plotGraph(actual, output, 'Actual position', 'Predicted position')
        util.plotLine(errors, 'Error graph') 
        print len(actual), len(output) 
        print "L2 error is >>>>>>>>"
        print math.sqrt(np.sum(errors))
        
        
        
        
    else:
    	sys.exit(1)
    
