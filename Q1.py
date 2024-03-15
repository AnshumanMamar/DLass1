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
wandb.init(project="DL-Assignment-1")

class NeuralNetwork:
    def __init__(self):
        self.w,self.b,self.a,self.h,self.wd,self.ad,self.hd,self.bd=[],[],[],[],[],[],[],[]


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
        elif act == 'identity':
            return k
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
        if act=='relu':
            relu_derivative=0
            relu_derivative=np.maximum(0,k)
            relu_derivative[relu_derivative>0]=1
            return relu_derivative
        if act=='tanh':
            a=np.exp(k)-np.exp(-k)
            b=np.exp(k)+np.exp(-k)
            x=a/b
            return 1-np.square(x)
        if act == 'identity':
            return 1

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
        # all_layer = [784,128,,,,,10]
        if start=='random':
            i =0
            while i <= hidden_layers:
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
        # print(len(self.w)-1,"check")
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
        return ((x2.shape[0]-n)/y2.shape[0])*100

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
        # labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        # wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(preds=ypred, y_true=y_test,class_names=labels)})
        acc=((x2.shape[0]-n)/y2.shape[0])*100
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

    #function to implement one combination of the forward and backward pass

    def onePass(self,x1,y1,classes,lay,rate,act,fn_loss,mom):
        self.forward_pass(x1 ,act)

        self.backward_pass(self.h[lay-1], y1,x1,classes, act,fn_loss,mom)

    #Function to implement stochastic gradient descent

    def batch(self,x1,y1,classes,lay,epo,count,size,act,fn_ans,mom):
        info,res=self.createBatches(x1,y1,size)

        i=0
        while i < epo:
            h=None
            j=0
            s=len(info)
            while j< s:
                self.onePass(info[j],res[j],classes,lay,count,act,fn_ans,mom)
                k=0
                while k< lay:
                    q=lay-1-k
                    self.w[k]-=count*(self.wd[q])
                    self.b[k]-=count*self.bd[q]
                    k+=1
                j+=1
            i+=1
            self.forward_pass(x1,act)
            loss_train=1
            loss1=self.loss_function(fn_ans,self.h[lay-1],y1,mom)
            self.forward_pass(x_val,act)
            loss2=self.loss_function(fn_ans,self.h[lay-1],y_val,mom)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            # wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Loss : "+str(loss1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validation Loss : "+str(loss2))
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Accurcy : "+str(acc1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validaion Accuracy: "+str(acc2))

    #Function to implement momentum based gradient descent

    def momentum(self,x1,y1,classes,lay,epo,count,size,eta,act,fn_loss,mom):
        info,res=self.createBatches(x1,y1,size)

        alpha,beta=[],[]
        i=0
        while i< lay:
            a=np.zeros((self.w[i].shape))
            b=np.zeros(self.b[i].shape)
            beta.append(b)
            alpha.append(a)
            i+=1
        i=0
        while i<epo :
            j=0
            while j< len(info):
                self.onePass(info[j],res[j],classes,lay,count,act,fn_loss,mom)
                k=0
                while k < lay:
                    s=lay-1-k
                    alpha[k]=(alpha[k]*eta)+self.wd[s]
                    beta[k]=(beta[k]*eta)+self.bd[s]
                    self.w[k]-=count*alpha[k]
                    self.b[k]-=count*beta[k]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            loss1=self.loss_function(fn_loss,self.h[lay-1],y1,mom)
            self.forward_pass(x_val,act)
            loss2=self.loss_function(fn_loss,self.h[lay-1],y_val,mom)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            # wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Loss : "+str(loss1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validation Loss : "+str(loss2))
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Accurcy : "+str(acc1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validaion Accuracy: "+str(acc2))

    #Function to implement nestrov gradient descent

    def nestrov(self,x1,y1,classes,lay,epo,count,size,eta,act,fn_loss,mom):
        info,res=self.createBatches(x1,y1,size)

        alpha,beta=[],[]

        i=0
        while i<lay:
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.w[i].shape))
            beta.append(b)
            alpha.append(a)
            i+=1

        i=0
        while i< epo:
            j=0
            while j< len(info):
                k=0
                while k < lay:
                    self.b[k]-=eta*beta[k]
                    self.w[k]-=eta*alpha[k]
                    k+=1
                self.onePass(info[j],res[j],classes,lay,count,act,fn_loss,mom)
                k=0
                while k<lay:
                    s=lay-1-k
                    alpha[k]=(eta*alpha[k])+count*(self.wd[s])
                    beta[k]=(eta*beta[k])+count*self.bd[s]
                    self.b[k]-=beta[k]
                    self.w[k]-=alpha[k]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.loss_function(fn_loss,self.h[s],y1,mom)
            self.forward_pass(x_val,act)
            loss2=self.loss_function(fn_loss,self.h[s],y_val,mom)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            # wandb.log({"train_accuracy":acc_train*100,"train_error":loss_train,"val_accuracy":acc_val*100,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Loss : "+str(loss1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validation Loss : "+str(loss2))
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Accurcy : "+str(acc1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validaion Accuracy: "+str(acc2))

    #Function to implement rmsProp gradient descent



    def rmsProp(self,x1,y1,classes,lay,epo,count,size,eta,act,fn_loss,mom,e):
        info,res=self.createBatches(x1,y1,size)

        alpha,beta=[],[]

        i=0
        while i< lay:
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.w[i].shape))
            beta.append(b)
            alpha.append(a)
            i+=1

        i=0
        while i < int(epo):
            j=0
            while j < len(info):
                self.onePass(info[j],res[j],classes,lay,count,act,fn_loss,mom)
                k=0
                while k < lay:
                    s=lay-1-k
                    q=1-eta
                    alpha[k]=(alpha[k]*eta)+(q)*np.square(self.wd[s])
                    beta[k]=(beta[k]*eta)+(q)*np.square(self.bd[s])
                    self.w[k]-=(count/np.sqrt(np.linalg.norm(alpha[k]+e)))*self.wd[s]
                    self.b[k]-=(count/np.sqrt(np.linalg.norm(beta[k]+e)))*self.bd[s]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.loss_function(fn_loss,self.h[s],y1,mom)
            self.forward_pass(x_val,act)
            loss2=self.loss_function(fn_loss,self.h[s],y_val,mom)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            # wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Loss : "+str(loss1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validation Loss : "+str(loss2))
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Accurcy : "+str(acc1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validaion Accuracy: "+str(acc2))

    #Function to implement adam gradient descent

    def adam(self,x1,y1,classes,lay,epo,count,size,eta1,eta2,act,fn_loss,e,mom):
        info,res=self.createBatches(x1,y1,size)

        w1,w2,b1,b2=[],[],[],[]

        i=0
        while i< lay:
            b=np.zeros((self.w[i].shape))
            a=np.zeros((self.w[i].shape))
            w2.append(b)
            w1.append(a)
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.b[i].shape))
            b1.append(a)
            b2.append(b)
            i+=1

        a=0
        i = 0
        while i < int(epo):
            j=0
            while j< len(info):
                a=a+1
                self.onePass(info[j],res[j],classes,lay,count,act,fn_loss,mom)
                k=0
                while k< lay:
                    r=1-eta1
                    p=1-eta2
                    u=1-eta1**a
                    v=1-eta2**a
                    s=lay-1-k

                    w1[k]=(w1[k]*eta1)+(r)*self.wd[s]
                    mwhat=w1[k]/(u)

                    w2[k]=(w2[k]*eta2)+(p)*np.square(self.wd[s])
                    vwhat=w2[k]/(v)

                    b1[k]=(b1[k]*eta1)+(r)*self.bd[s]
                    mbhat=b1[k]/(u)

                    b2[k]=(b2[k]*eta2)+(p)*np.square(self.bd[s])
                    vbhat=b2[k]/(v)

                    self.w[k]-=(count/np.sqrt(vwhat+e))*mwhat
                    self.b[k]-=(count/np.sqrt(vbhat+e))*mbhat
                    k+=1
                j+=1
            i=i+1
            self.forward_pass(x1, act)
            s=lay-1
            loss1=self.loss_function(fn_loss,self.h[s],y1,mom)
            self.forward_pass(x_val,act)
            loss2=self.loss_function(fn_loss,self.h[s],y_val,mom)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            # wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            print("Iteration Number: "+str(i), end="")
            print(" Train Loss : "+str(loss1))
            print("Iteration Number: "+str(i), end="")
            print(" Validation Loss : "+str(loss2))
            print("Iteration Number: "+str(i), end="")
            print(" Train Accurcy : "+str(acc1))
            print("Iteration Number: "+str(i), end="")
            print(" Validaion Accuracy: "+str(acc2))

    #Function to implement Nadam Gradient descent

    def Nadam(self,x1,y1,classes,lay,epo,count,size,eta1,eta2,act,fn_loss,e,mom):
        w1,w2,b1,b2=[],[],[],[]
        info,res=self.createBatches(x1,y1,size)

        for i in range(lay):
            b=np.zeros((self.w[i].shape))
            a=np.zeros((self.w[i].shape))
            w2.append(b)
            w1.append(a)
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.b[i].shape))
            b1.append(a)
            b2.append(b)

        a=0
        i=0
        while i< int(epo):
            j=0
            while j< len(info):
                a=a+1
                self.onePass(info[j],res[j],classes,lay,count,act,fn_loss,mom)
                k=0
                while k < lay:
                    r=1-eta1
                    p=1-eta2
                    u=1-eta1**a
                    v=1-eta2**a
                    s=lay-1-k
                    w1[k]=(w1[k]*eta1)+(r)*self.wd[s]
                    mwhat=w1[k]/(u)

                    w2[k]=(w2[k]*eta2)+(p)*np.square(self.wd[s])
                    vwhat=w2[k]/(v)

                    b1[k]=(b1[k]*eta1)+(r)*self.bd[s]
                    mbhat=b1[k]/(u)

                    b2[k]=(b2[k]*eta2)+(p)*np.square(self.bd[s])
                    vbhat=b2[k]/(v)

                    self.w[k]-=(count/np.sqrt(vwhat+e))*(eta1*mwhat+(((r)*self.wd[s])/(u)))
                    self.b[k]-=(count/np.sqrt(vbhat+e))*(eta1*mbhat+(((r)*self.bd[s])/(u)))
                    k=k+1
                j=j+1
            i=i+1
            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.loss_function(fn_loss ,self.h[s],y1,mom)
            self.forward_pass(x_val ,act)
            loss2=self.loss_function(fn_loss,self.h[s], y_val,mom)
            acc1=self.accuracy(x1 ,y1 ,act)
            acc2=self.accuracy(x_val ,y_val,act)
            # wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            wandb.log(
                    {
                        'epoch': i,
                        'training_loss' : round(loss1,2),
                        'training_accuracy' : round(acc1,2),
                        'validation_loss' : round(loss2,2),
                        'validation_accuracy':round(acc2,2)
                    }
                )
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Loss : "+str(loss1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validation Loss : "+str(loss2))
            # print("Iteration Number: "+str(i), end="")
            # print(" Train Accurcy : "+str(acc1))
            # print("Iteration Number: "+str(i), end="")
            # print(" Validaion Accuracy: "+str(acc2))


    #Main Function to implement the functionality

    def architecture(self,x1,y1,x2,y2,classes,hiddenlayers,neuron,input_neuron,batch,initialization,fn_loss,activation,optimizer,n,iter,beta,beta1,beta2,e,alpha,mom):
        self.w,self.b=[],[]
        self.make_layers(hiddenlayers,neuron,input_neuron,initialization,classes)
        # print(len(self.w),"arch")
        if optimizer=="batch":
            self.batch(x1,y1,classes,len(self.w),iter,n,batch,activation,fn_loss,alpha)
        elif optimizer=='momentum':
            self.momentum(x1,y1,classes,len(self.w),iter,n,batch,mom,activation,fn_loss,alpha)
        elif optimizer=='nestrov':
            self.nestrov(x1,y1,classes,len(self.w),iter,n,batch,beta,activation,fn_loss,alpha)
        elif optimizer=='rmsProp':
            self.rmsProp(x1,y1,classes,len(self.w),iter,n,batch,beta,activation,fn_loss,alpha,e)
        elif optimizer=='adam':
            self.adam(x1,y1,classes,len(self.w),iter,n,batch,beta1,beta2,activation,fn_loss,e,alpha)
        elif optimizer=='Nadam':
            self.Nadam(x1,y1,classes,len(self.w),iter,n,batch,beta1,beta2,activation,fn_loss,e,alpha)

#creating the object and calling


obj=NeuralNetwork()
obj.architecture(x1,
                 y1,
                 x_val,
                 y_val,
                 classes=10,
                 hiddenlayers=5,
                 neuron=128,
                 input_neuron=784,
                 batch=64,
                 initialization="xavier",
                 fn_loss="cross_entropy",
                 activation="relu",
                 optimizer="adam",
                 n=0.001,
                 iter=10,
                 beta=0.5,
                 beta1=0.999,
                 beta2=0.999,
                 e=1e-6,
                 alpha=0,
                 mom=0.9)
obj.predict(x2,y2,act="relu")
