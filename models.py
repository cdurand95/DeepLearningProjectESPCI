import torch as th
import torch.nn.functional as F

def define_model(name, image_size_x, image_size_y, n_sensors,minibatch_size, kernel_size=4):
    if name == 'with_deconvolution':
        return model_with_deconv(image_size_x, image_size_y, n_sensors,minibatch_size,kernel_size)

    elif name == 'without_deconvolution':
        return model_without_deconv(image_size_x, image_size_y, n_sensors,minibatch_size)

    raise ValueError('model {} not recognized'.format(name))


class model_with_deconv(th.nn.Module):

    def __init__(self, image_size_x, image_size_y, n_sensors, minibatch_size, kernel_size):

        super(model_with_deconv, self).__init__()

        self.minibatch_size=minibatch_size
        self.Din=n_sensors
        self.Hout=image_size_x
        self.Wout=image_size_y
        self.kernelsize=kernel_size

        self.stride=(2,2)
        self.padding=(1,1)

        if ((self.Hout-self.kernelsize+2*self.padding[0])%self.stride[0])==0:
            self.outputpaddingx=0
        else :
            self.outputpaddingx=1
        if ((self.Wout-self.kernelsize+2*self.padding[1])%self.stride[1])==0:
            self.outputpaddingy=0
        else :
            self.outputpaddingy=1

        self.Hin=int(1 + (self.Hout-self.kernelsize+2*self.padding[0]-self.outputpaddingx)/self.stride[0])
        self.Win=int(1 + (self.Wout-self.kernelsize+2*self.padding[1]-self.outputpaddingy)/self.stride[1])

        self.linear1 = th.nn.Linear(self.Din, 40)
        self.norm1= th.nn.BatchNorm1d(1)
        self.linear2 = th.nn.Linear(40, 45)
        self.norm2= th.nn.BatchNorm1d(1)
        self.linear3 = th.nn.Linear(45, self.Hin*self.Win)
        self.deconv=th.nn.ConvTranspose2d(1, 1, kernel_size=self.kernelsize, stride=self.stride, padding=self.padding, output_padding=(self.outputpaddingx, self.outputpaddingy))
        self.outputfct=th.nn.Tanh()


    def forward(self, input):
        x=F.relu(self.linear1(input))
        x = self.norm1(x.view(self.minibatch_size,1,-1))
        x=x.view(self.minibatch_size,40)
        x=F.relu(self.linear2(x))
        x = self.norm2(x.view(self.minibatch_size,1,-1))
        x=x.view(self.minibatch_size,45)
        x = self.linear3(x)
        x=x.view(self.minibatch_size,1,self.Hin ,self.Win)
        x=self.deconv(x)
        x=x.view(self.minibatch_size,self.Hout*self.Wout)
        x=self.outputfct(x)
        return x


class model_without_deconv(th.nn.Module):

    def __init__(self, image_size_x, image_size_y, n_sensors, minibatch_size):

        super(model_without_deconv, self).__init__()

        self.minibatch_size=minibatch_size
        self.D_in=n_sensors
        self.D_out=image_size_x*image_size_y

        self.linear1 = th.nn.Linear(self.D_in, 40)
        self.norm1= th.nn.BatchNorm1d(1)
        self.linear2 = th.nn.Linear(40, 45)
        self.norm2= th.nn.BatchNorm1d(1)
        self.linear3 = th.nn.Linear(45, self.D_out)
        self.outputfct=th.nn.Tanh()


    def forward(self, input):
        x=F.relu(self.linear1(input))
        x = self.norm1(x.view(self.minibatch_size,1,-1))
        x=x.view(self.minibatch_size,40)
        x=F.relu(self.linear2(x))
        x = self.norm2(x.view(self.minibatch_size,1,-1))
        x=x.view(self.minibatch_size,45)
        x = self.outputfct(self.linear3(x))
        return x
