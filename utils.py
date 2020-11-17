import numpy as np
from scipy.io import loadmat
import torch as th
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt

#Function to define the coordinates in pixels of sensors from an array Theta containing the angles in degrees
#of positionning of sensors around the cylinder
def define_sensors_coordinates_from_theta(Theta):
    '''From an array of values of theta in °, returns an array of x and y coordinates'''
    X_coordinates=np.round(36 * np.cos(Theta)) + 0; #cylinder is 36 pixels wide in radius
    Y_coordinates=np.round(36 * np.sin(Theta)) + 99;
    Sensors_coordinates=[np.int64(X_coordinates),np.int64(Y_coordinates)]
    return Sensors_coordinates

#Function to select n_sensors random positions for sensors
def random_sensors_positionning(n_sensors,random_seed):
    '''Returns an array of n_sensors x and y coordinates ranbdomly generated from the random seed'''
    random.seed(random_seed) #For reproducibility
    randomlist = random.sample(range(-45, 45), n_sensors) #all sensors will be separeted by at least 2 degrees
    Theta=2*np.array(randomlist)*2*np.pi/360;
    return define_sensors_coordinates_from_theta(Theta)

#Function to transform data into pytorch tensors suitable for the training of the model
def data_treatment(Data, Training_size, Sensors_coordinates):
    '''Takes into argument the raw data, the desired size for the training set, and the sensors coordinates.
    Returns extracted input values for training and validation set, the normalized linearized tensor data set for training set and validation set, and the value of the norm used'''

    (n,m,l)=Data.shape;

    #Separating data into training and validation set
    Training_data=Data[0:Training_size,:,:];
    Validation_data=Data[Training_size:151,:,:];

    #Creating input arrays from sensors coordinates
    Input_training=Training_data[:,Sensors_coordinates[0],Sensors_coordinates[1]];
    Input_validation=Validation_data[:,Sensors_coordinates[0],Sensors_coordinates[1]];

    #Transforming data into pytorch tensors
    Input_training=th.FloatTensor(Input_training);
    Input_validation=th.FloatTensor(Input_validation);
    Training_set=th.FloatTensor(Training_data);
    Validation_set=th.FloatTensor(Validation_data);

    #Reshaping data into linearized tensors
    Training_set=Training_set.view(Training_size,m*l);
    Validation_set=Validation_set.view(n-Training_size,m*l);

    #Normalizing data to hav data between -1 and 1
    Vmax=th.max(Training_set);
    Vmin=th.min(Training_set);
    Norm=th.max(Vmax,abs(Vmin));
    epsilon=0.02;
    Norm=Norm+epsilon;
    Training_set=Training_set/Norm;
    Validation_set=Validation_set/Norm;

    return Input_training, Input_validation, Training_set, Validation_set, Norm

#Function to train the model
def train_model(Model, Input_training, Input_validation, Training_set, Validation_set, learning_rate, weight_decay, nepochs, minibatch_size):
    '''Trains the model over n_epochs epochs with MSE loss and Adam optimizer. Prints regularly the evolution of loss and error.
    Returns the trained model, the loss and the error during training, and the computing time'''
    # Defining loss function and optimizer
    loss_fn = th.nn.MSELoss()
    optimizer = th.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay = weight_decay )

    #Defining arrays to store values of loss and error
    Loss_training=np.zeros(nepochs);
    Error_validation=np.zeros(nepochs);

    # Mini-batching
    Training_size=Training_set.shape[0];
    idx = np.arange(Training_size) #index of training ssequence
    nbatch = int(Training_size/minibatch_size)

    #Training
    fprint=nepochs/10 #frequency to print follow up of training
    tic=time.perf_counter(); #for timing
    for epoch in range(nepochs):
        total_loss=0;
        np.random.shuffle(idx)

        for k in range(nbatch-1):
            ids=idx[k*minibatch_size:(k+1)*minibatch_size]
            prediction=Model.forward(Input_training[ids,:].squeeze());
            loss = loss_fn(prediction, Training_set[ids,:].squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss
        Loss_training[epoch]=total_loss.detach().squeeze().numpy()


        #Validation sequence (only on minibatchsize)
        prediction=Model.forward(Input_validation[0:minibatch_size,:].squeeze());
        error_valid=0
        for m in range(minibatch_size):
            error_valid=error_valid+np.sqrt(np.sum(np.square(Validation_set[m,:].squeeze().detach().numpy()-prediction[m,:].squeeze().detach().numpy()))/np.sum(np.square(Validation_set[m,:].squeeze().detach().numpy())))
        error_valid=error_valid/minibatch_size
        Error_validation[epoch]=error_valid;

        if epoch%fprint==0:
            toc=time.perf_counter();
            print('Epoch=', epoch,', Loss=',total_loss.detach().numpy(), ', Error= ',error_valid, ', Time since beginning of training :', toc-tic, ' seconds')

    toc=time.perf_counter();
    timing=toc-tic;

    return Model, Loss_training, Error_validation, timing

