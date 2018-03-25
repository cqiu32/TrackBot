import numpy as np

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """





    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))
                      

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    
    '''

    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  
    K = P * H.T * S.I    
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) 
    P = (I - K*H)*P


    x = F*x + motion
    P = F*P*F.T + Q

    return x, P
    
def kalman_prediction(measurements):
    R = [[1]]
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty
    result = []
    predicted_moves=[]
    
    for meas in measurements:
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    
    kalman_x, kalman_y = zip(*result)
    x_eta  = kalman_x[-1][0]
    y_eta  = kalman_y[-1][0]
    temp = measurements
    temp.append([x_eta,y_eta])
    
    for i in range(NumberOfPredictions):
        R = [[1]]
        x = np.matrix('0. 0. 0. 0.').T 
        P = np.matrix(np.eye(4))*1000 # initial uncertainty
        result = []
        for meas in temp:
            x, P = kalman_xy(x, P, meas, R)
            result.append((x[:2]).tolist())
        
        kalman_x, kalman_y = zip(*result)    
        temp.pop(0)
        temp.append([kalman_x[-1][0],kalman_y[-1][0]])
        predicted_moves.append([kalman_x[-1][0],kalman_y[-1][0]])
    

    return x_eta,y_eta,predicted_moves

# need to predict for 60 frames or 2 seconds.
NumberOfPredictions = 60
f = open('training_data.txt');
data =np.loadtxt(f, delimiter = ',');

motions = []

for i in range(len(data)):
    motions.append(data[i]);

