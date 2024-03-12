# !pip install wandb

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
# import wandb
from sklearn.model_selection import train_test_split
import math


# !wandb login f63b589f2c31fd6562d752168e172d22870ab562

# wandb.init(project="DL-Assignment-1", name="Q")

# x_train=None
# x_test=None
# y_train=None
# y_test=None

#Loading the dataset
# if dataset=="fashion_mnist":
# from keras.datasets import fashion_mnist
(x1, y1), (x2, y2) = fashion_mnist.load_data()
# elif dataset=="mnist":
#     from keras.datasets import mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshaping and normalizing the dataset
x1=x1.reshape(x1.shape[0],(x1.shape[1]*x1.shape[2]))
x2=x2.reshape(x2.shape[0],(x2.shape[1]*x2.shape[2]))
x2=x2/255
x1=x1/255


#create validation dataset
x1, x_val, y1, y_val = train_test_split(x1,y1, test_size=0.1, random_state=0)

class NeuralNetwork:
    def __init__(self):
        self.w,self.b,self.a,self.h,self.wd,self.ad,self.hd,self.bd=[],[],[],[],[],[],[],[]
        # self.b=[]
        # self.a=[]
        # self.h=[]
        # self.wd=[]
        # self.ad=[]
        # self.hd=[]
        # self.bd=[]

    #collection of all the activation function implementation

    def activations(self,act,k):
        if act=='sigmoid':
            s=1+np.exp(-k)
            return 1/s
        elif act =='relu':
            s=k*(k>0)
            return s
        elif act =='tanh':
            s = np.tanh(k)
            return s
        elif act =='softmax':
            x=np.copy(k)
            i=0
            while i < k.shape[0]:
                add=0
                largi=np.argmax(k[i])
                j=0
                while j< k.shape[1]:
                    add+=np.exp(k[i][j]-k[i][largi])
                    j=j+1
                k[i]=np.exp(k[i]-k[i][largi])/add
                x[i]=k[i]
                i=i+1
            return x

    #collection of all the activation function derivative

    def activations_derivative(self,act,k):
        if act=='sigmoid':
            s = np.multiply((1/(1+np.exp(-k))),(1-(1/(1+np.exp(-k)))))
            return s
        elif act=='relu':
            relu_derivative=0
            relu_derivative=np.maximum(0,k)
            relu_derivative[relu_derivative>0]=1
            return relu_derivative
        elif act=='tanh':
            a=np.exp(k)-np.exp(-k)
            b=np.exp(k)+np.exp(-k)
            x=a/b
            return 1-np.square(x)

    #collection of all the loss function implementation

    def loss_function(self,fn_loss,yhat,y1,mom):
        cac=0
        i=0
        while i<len(self.w):
            s=np.sum(self.w[i]**2)
            cac=cac+s
            i=i+1
        cac=mom*cac
        cac=cac/2
        ch=1
        if fn_loss=='cross_entropy':
            loss=0
            i=0
            while i<y1.shape[0]:
                loss=loss-((np.log2(yhat[i][y1[i]])))
                i=i+1
            s=loss+cac
            return s/y1.shape[0]

        if fn_loss=='mean_square':
            bl=np.zeros((y1.shape[0],yhat.shape[1]))
            i=0
            while i<y1.shape[0]:
                bl[i][y1[i]]=1
                i=i+1
            s=(np.sum(((yhat-bl)**2)))+cac
            return s/(y1.shape[0])

    #make layer function is used to implement the two initialization of weight initialization i.e xavier and random

        # self.make_layers(no_of_hidden_layers,no_of_neuron,input_neuron,initialization,no_of_classes)

    def make_layers(self,hidden_layers,neuron,neuron_input,start,classes):

        all_layer=[]
        np.random.seed(5)
        all_layer.append(neuron_input)
        i=0
        while i<hidden_layers:
            all_layer.append(neuron)
            i=i+1
        intialization = 0
        all_layer.append(classes)
        if start=='random':
            i =0
            while i<= hidden_layers:
                wt=np.random.uniform(-0.5,0.5,(all_layer[i],all_layer[i+1]))
                b=np.random.uniform(-0.5,0.5,(1,all_layer[i+1]))
                self.b.append(b)
                self.w.append(wt)
                i=i+1
        if start=='xavier':
            i=0
            while i<= hidden_layers:
                n=np.sqrt(6/(all_layer[i]+all_layer[i+1]))
                wt=np.random.uniform(-n,n,(all_layer[i],all_layer[i+1]))
                b=np.random.uniform(-n,n,(1,all_layer[i+1]))
                self.b.append(b)
                self.w.append(wt)
                i=i+1
    #collection of the forward pass of feed forward neural network

    def forward_pass(self,x,act):

        check=x
        self.a,self.h=[],[]
        tp=act
        i=0
        while i<len(self.w)-1:
            a1=np.add(np.matmul(check,self.w[i]),self.b[i])
            bool = act=='relu' and i==0
            if (bool):
                j=0
                s=a1.shape[0]
                while j < s:
                    maxi=np.argmax(a1[j])
                    a1[j]/=a1[j][maxi]
                    j=j+1
            h1=self.activations(act,a1)
            check=h1
            self.h.append(h1)
            self.a.append(a1)
            i=i+1
        a1=np.add(np.matmul(check,self.w[len(self.w)-1]),self.b[len(self.w)-1])
        h1=self.activations('softmax',a1)
        self.h.append(h1)
        self.a.append(a1)


    #implementation of the backward propagation algorithm. The derivative is calculated based on the loss function

    def backward_pass(self,yhat,y1,x1,classes,activation,fn_loss,mom):
        self.wd,self.bd,self.ad,self.hd=[],[],[],[]
        # self.bd=[]
        # self.ad=[]
        # self.hd=[]
        bl=0
        bl=np.zeros((y1.shape[0],classes))
        i=0
        while i< y1.shape[0]:
            bl[i][y1[i]]=1
            i=i+1

        b=None
        a=None


        if fn_loss=="cross_entropy":
            yhatl=np.zeros((yhat.shape[0],1))
            i=0
            while i< yhat.shape[0]:
                yhatl[i]=yhat[i][y1[i]]
                i=i+1

            b=-1*(bl-yhat)
            a=-1*(bl/yhatl)

            self.hd.append(a)
            self.ad.append(b)

        if fn_loss=="mean_square":
            s=yhat-bl
            a=2*s
            self.hd.append(a)
            b=[]
            j=0
            while j< yhat.shape[1]:
                r=yhat.shape[0]
                s=yhat.shape[1]
                hot_j=np.zeros((r,s))
                hot_j[:,j]=1
                hat_j=np.ones((r,s))*(yhat[:,j].reshape(r,1))
                l=yhat-bl
                x=hot_j-hat_j
                aj=2*(np.sum((l)*(yhat*(x)),axis=1))
                b.append(aj)
                j=j+1
            self.ad.append(np.array(b).T)



        j = len(self.w)-1
        while j>-1:
            u=self.h[j-1].T
            bool = (j==0)
            if bool:
                u=x1.T
            t=x1.shape[0]
            length=len(self.ad)
            w=np.matmul(u,self.ad[-1])/t
            b=np.sum(self.ad[length-1],axis=0)/t
            if j!=0:
                a=np.matmul(self.ad[length-1],self.w[j].T)
                der=self.activations_derivative(activation,self.a[j-1])
                z=np.multiply(a,der)
                self.hd.append(a)
                self.ad.append(z)
            self.bd.append(b)
            self.wd.append(w)
            j=j-1
        i=0
        while i< len(self.w):
            self.wd[len(self.w)-1-i]-=mom*self.w[i]
            i=i+1
    #Function to calculate accuracy of the dataset

    def accuracy(self,x2,y2,act):
        self.forward_pass(x2,act)
        ypred=np.argmax(self.h[len(self.w)-1],axis=1)
        n=0
        l=y2.shape[0]
        i=0
        while i<y2.shape[0]:
            if ypred[i]!=y2[i]:
                n+=+1
            i+=1
        return ((x2.shape[0]-n)/y2.shape[0])

    #This function is used to make the final prediction on the test data

    def predict(self,x2,y2,act):
        self.forward_pass(x2,act)
        fut=np.argmax(self.h[len(self.w)-1],axis=1)
        n=0
        i=0
        while i < y2.shape[0]:
            if fut[i]!=y2[i]:
                n+=1
            i+=1

        #confusion matrix
        labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        # wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(preds=ypred, y_true=y_test,class_names=labels)})
        acc=((x2.shape[0]-n)/y2.shape[0])
        print("Test Accuracy: "+str(acc))

    #Function to create the batches out of the original data

    def createBatches(self,x1,y1,size):
        info,res=[],[]
        s=x1.shape[0]
        l=math.ceil(s/size)
        i=0
        while i < l:
            group,group_ans=[],[]
            j=i*size
            s=min((i+1)*size,x1.shape[0])
            while j< s:
                group.append(x1[j])
                group_ans.append(y1[j])
                j+=1
            group_ans=np.array(group_ans)
            group=np.array(group)
            info.append(group)
            res.append(group_ans)
            i+=1
        return info,res
