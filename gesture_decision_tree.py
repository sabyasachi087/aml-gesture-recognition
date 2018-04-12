import numpy as np


class HandGestureDecider:
    
    def __init__(self, pt=75):
        """ @param 
            pt (in %) -> Probability Threshold, Minimum matching probability
            for deciding on the gesture. Default value is 75%
        """
        self.probability_threshold = pt
    
    def probability(self, gestures, distances, prob_map):
        for indx in range(gestures):
            gest = gestures[indx]
            if gest in prob_map.keys():
                prob_map[gest]
                
    
    def decide(self, gyro, acc, fingers):
        """@param
        gyro,acc,fingers -> Gyroscope / Accelerometer / Fingers DTW data as tuple with first index having list of gestures 
        and the second index should have the dtw distance values for each of the gestures 
        """
        prob_map = dict()
        for indx in range(len(gyro)):
            gestures, dist = gyro[indx]
