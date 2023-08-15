import numpy as np

class NeuralNet:

    def __init__(self,LayerStruct,ActivationType):
        self.depth = np.size(LayerStruct)-1
        self.weight = np.empty(self.depth,dtype=object)
        self.bias = np.empty(self.depth,dtype=object)
        self.ActivationType = ActivationType
        self.FirstMomentW = np.empty(self.depth,dtype=object)
        self.FirstMomentB = np.empty(self.depth,dtype=object)
        self.SecondMomentW = np.empty(self.depth,dtype=object)
        self.SecondMomentB = np.empty(self.depth,dtype=object)
        self.LayerStruct = LayerStruct
        self.LayerNumOfW = np.empty(self.depth,dtype=object)
        self.LayerNumOfB = np.empty(self.depth,dtype=object)

        NumOfParameters=0
        for i in range(np.size(LayerStruct)-1):
            NumOfLocalWeight=LayerStruct[i]*LayerStruct[i+1]
            NumOfParameters=NumOfParameters+NumOfLocalWeight
            self.LayerNumOfW[i]=NumOfLocalWeight
            self.LayerNumOfB[i]=LayerStruct[i+1]

        NumOfParameters=NumOfParameters+np.sum(LayerStruct)-LayerStruct[0]
        self.NumOfWeight=np.sum(self.LayerNumOfW)
        self.NumOfBias=np.sum(self.LayerNumOfB)
        self.NumOfParameters=NumOfParameters


        for i in range(self.depth):
            self.weight[i]=np.random.uniform(low=-1, high=1, size=(LayerStruct[i+1],LayerStruct[i]))
            self.bias[i]=np.random.uniform(low=-1, high=1, size=(LayerStruct[i+1],1))
            self.FirstMomentW[i]=np.zeros((LayerStruct[i+1],LayerStruct[i]))
            self.FirstMomentB[i]=np.zeros((LayerStruct[i+1],1))
            self.SecondMomentW[i]=self.FirstMomentW[i]
            self.SecondMomentB[i]=self.FirstMomentB[i]

    def activation(self,x):
        if self.ActivationType=="tanh":
            a=np.tanh(x)
        elif self.ActivationType=="gaussian":
            a=np.exp(-x**2)
        elif self.ActivationType=="sigmoid":
            a=1/(1+np.exp(-x))
        elif self.ActivationType=="relu":
            a=x*np.sign(x)
        return a

    def ActivationDerivate(self,z,a):
        if self.ActivationType=="tanh":
            d=1-a**2
        elif self.ActivationType=="gaussian":
            d=-2*z*a
        elif self.ActivationType=="sigmoid":
            d=a*(1-a)
        elif self.ActivationType=="relu":
            d=z>0
        return d

    def Evaluate(self,x):
        v=x
        for i in range(self.depth-1):
            v=self.weight[i] @ v + self.bias[i]
            v=self.activation(v)
        fx=self.weight[-1] @ v+self.bias[-1]
        return fx
    
def AutomaticGradient(x,y,Net):
    A=np.empty(Net.depth,dtype=object)
    D=np.empty(Net.depth,dtype=object)
    v=x

    for i in range(Net.depth-1):
        z=Net.weight[i] @ v+Net.bias[i]
        v=Net.activation(z)
        A[i]=v
        D[i]=Net.ActivationDerivate(z,v)
    z=Net.weight[-1] @ v+Net.bias[-1]
    A[-1]=z; D[-1]=z
    ErrorVector=(z-y)/y.size
    g=ErrorVector;
    dw=np.empty(Net.depth,dtype=object)
    db=np.empty(Net.depth,dtype=object)

    dw[-1]=g @ (np.transpose(A[-2]))
    S=np.sum(g,keepdims=True)

    db[-1]=S
    for i in range(Net.depth-2,0,-1):
        g=D[i]*(np.transpose(Net.weight[i+1]) @ g)
        Ap=np.transpose(A[i-1])
        dw[i]=g @ Ap
        db[i]=np.sum(g,1,keepdims=True)

    g=D[0]*(np.transpose(Net.weight[1]) @ g)
    Ap=np.transpose(x)
    dw[0]=g @ Ap
    db[0]=np.sum(g,1,keepdims=True)

    return dw, db

def CostFunction(x,y,NN):
    p=NN.Evaluate(x); e=y-p
    J=np.mean(e**2)
    return J

def ADAM(grad,FirstMoment,SecondMoment):
        m1=0.9; m2=0.999; epsilon=1e-8
        FirstMoment=m1*FirstMoment+(1-m1)*grad
        SecondMoment=m2*SecondMoment+(1-m2)*(grad**2)
        d=FirstMoment/(np.sqrt(SecondMoment)+epsilon)

        return d , FirstMoment ,SecondMoment

def SGDM(grad,FirstMoment):
    m1=0.9
    FirstMoment=m1*FirstMoment+(1-m1)*grad
    d=FirstMoment

    return d , FirstMoment

def Shuffle(x,y,BatchSize):
    NumOfData=y.size
    NumOfBatch=int(np.floor(NumOfData/BatchSize)+1)
    Remained=np.remainder(NumOfData,BatchSize)

    Xdata=np.empty(NumOfBatch,dtype=object)
    Ydata=np.empty(NumOfBatch,dtype=object)
    Index=np.arange(NumOfData); np.random.shuffle(Index)

    for i in range(NumOfBatch-1):
        Xdata[i]=x[:,Index[i*BatchSize:(i+1)*BatchSize]]
        Ydata[i]=y[:,Index[i*BatchSize:(i+1)*BatchSize]]


    if Remained==0:
        Xdata[NumOfBatch-1]=x[:,-Remained-1:-1]
        Ydata[NumOfBatch-1]=y[:,-Remained-1:-1]
    else:
        Start=(NumOfBatch-1)*BatchSize
        Xdata[NumOfBatch-1]=x[:,Index[Start:NumOfData]]
        Ydata[NumOfBatch-1]=y[:,Index[Start:NumOfData]]

    return Xdata,Ydata

def FirstOrderSolver(x,y,solver,MaxIter,StepSize,BatchSize,Net):

    # Assign Update Rule
    if solver=="ADAM":
        def UpdateMethod(dw,db,Net):
            for j in range(Net.depth):
                DescentW, Net.FirstMomentW[j], Net.SecondMomentW[j] = ADAM(dw[j],Net.FirstMomentW[j],Net.SecondMomentW[j])
                DescentB, Net.FirstMomentB[j], Net.SecondMomentB[j] = ADAM(db[j],Net.FirstMomentB[j],Net.SecondMomentB[j])
                Net.weight[j]=Net.weight[j]-StepSize*DescentW
                Net.bias[j]=Net.bias[j]-StepSize*DescentB
            UpdatedNet=Net
            return UpdatedNet
    elif solver=="SGDM":
        def UpdateMethod(dw,db,Net):
            for j in range(Net.depth):
                DescentW, Net.FirstMomentW[j] = SGDM(dw[j],Net.FirstMomentW[j])
                DescentB, Net.FirstMomentB[j] = SGDM(db[j],Net.FirstMomentB[j])
                Net.weight[j]=Net.weight[j]-StepSize*DescentW
                Net.bias[j]=Net.bias[j]-StepSize*DescentB
            UpdatedNet=Net
            return UpdatedNet

    

    CostList=np.zeros((MaxIter,1))
    DispNum=np.floor(MaxIter/10)
    # Iterations
    
    NumOfData=y.size
    NumOfBatch=int(np.floor(NumOfData/BatchSize)+1)
    for i in range(MaxIter):

        if BatchSize!=NumOfData:
            sx, sy = Shuffle(x,y,BatchSize)

        for j in range(NumOfBatch):
            dw, db = AutomaticGradient(sx[j],sy[j],Net)
            Net=UpdateMethod(dw,db,Net)

        C=CostFunction(x,y,Net)
        CostList[i]=C
        if np.mod(i,DispNum)==0:
            print("Iteration: {}, Cost: {:4.4f}".format(i+1, C))

    error=y-Net.Evaluate(x)
    MAE=np.mean(abs(error))
    print("Max Iteration: {}, Mean Absolute Error: {:4.4f}".format(MaxIter, MAE))
    TrainedNet=Net

    return TrainedNet

def GradMatToVec(MatrixW,MatrixB,Net):
    WShape=np.empty(Net.depth,dtype=object)
    BShape=np.empty(Net.depth,dtype=object)
    # Initialize
    w=MatrixW[0]; b=MatrixB[0]
    WShape[0]=w.shape; BShape[0]=b.shape
    w=np.array(w.flatten()); sw=w
    b=np.array(b.flatten()); sb=b
    for i in range(Net.depth-1):
        w=MatrixW[i+1]; b=MatrixB[i+1]
        WShape[i+1]=w.shape; BShape[i+1]=b.shape
        w=np.array(w.flatten()); b=np.array(b.flatten())
        sw=np.concatenate([sw,w])
        sb=np.concatenate([sb,b])

        Vector=np.concatenate([sw,sb])
        Vector=np.array([Vector])
        Vector=np.transpose(Vector)
    return Vector

def MatrixToVector(Net):
    WShape=np.empty(Net.depth,dtype=object)
    BShape=WShape
        # Initialize
    w=Net.weight[0]; b=Net.bias[0]
    WShape[0]=w.shape; BShape[0]=b.shape
    w=np.array(w.flatten()); sw=w
    b=np.array(b.flatten()); sb=b
    for i in range(Net.depth-1):
        w=Net.weight[i+1]; b=Net.bias[i+1]
        WShape[i+1]=w.shape
        BShape[i+1]=b.shape
        w=np.array(w.flatten())
        b=np.array(b.flatten())
        sw=np.concatenate([sw,w])
        sb=np.concatenate([sb,b])
        
        Vector=np.concatenate([sw,sb])
        Vector=np.array([Vector])
        Vector=np.transpose(Vector)
    return Vector

def VectorToMatrix(Vector,Net):

        # VectorW=Vector[0:Net.NumOfWeight]
    VectorB=Vector[Net.NumOfWeight:Net.NumOfParameters]
    MatrixW=np.empty(Net.depth,dtype=object)
    MatrixB=np.empty(Net.depth,dtype=object)
    CounterW=0; CounterB=0
    for i in range(Net.depth):
        # Weight Reshape
        EndIndexW=CounterW+Net.LayerNumOfW[i]
        VW=Vector[CounterW:EndIndexW]
        Wshape=np.shape(Net.weight[i])
        MatrixW[i]=np.reshape(VW,Wshape)
        CounterW=CounterW+Net.LayerNumOfW[i]
        # Bias Reshape
        EndIndexB=CounterB+Net.LayerNumOfB[i]
        MatrixB[i]=VectorB[CounterB:EndIndexB]
        CounterB=CounterB+Net.LayerNumOfB[i]
    return MatrixW, MatrixB

def BFGS(s,y,H):
    # Quasi-Newton method for CNN, Yi-ren, Goldfarb, 2022
    mu1=0.2; mu2=0.001
    Quad=np.transpose(y) @ H @ y
    InvRho=np.transpose(s) @ y
    if InvRho<mu1*Quad:
        theta=(1-mu1)*Quad/(Quad-InvRho)
    else:
        theta=1
    s=theta*s+(1-theta)*(H @ y); y=y+mu2*s
    Rho=1/(np.transpose(s) @ y); st=np.transpose(s); yt=np.transpose(y);
    H = H+(Rho**2)*(st @ y+ yt @ H @ y)*(np.outer(s, s))-Rho*(H @ np.outer(y, s) + np.outer(s, y) @ H)
    return H

def LineSearch(x,y,SearchVector,dp,Net):
        # Simple Backtracking Line Search
        C0=CostFunction(x,y,Net)
        MaxIterLS=20; c=1e-4; Decay=0.5
        p=MatrixToVector(Net)
        TempNN=Net
        Scalar=np.transpose(SearchVector) @ dp
        for j in range(MaxIterLS):
            step=np.power(Decay,j)
            pstar=p+step*SearchVector
            TempNN.weight, TempNN.bias=VectorToMatrix(pstar,TempNN)
            Cj=CostFunction(x,y,TempNN)
            LHS=Cj; RHS=C0+c*Scalar*step
            WolfeCondition=LHS<=RHS
            if WolfeCondition==1:
                DescentVector=step*SearchVector
                break

        return DescentVector,C0

def QuasiNewtonSolver(x,y,Net,MaxIter):
    

    # BFGS Iterations
    dw0, db0 = AutomaticGradient(x,y,Net)
    p=MatrixToVector(Net)
    dp=GradMatToVec(dw0,db0,Net)
    delta=1e-2
    H=delta*np.eye(Net.NumOfParameters)

    CostList=np.zeros((1,MaxIter))
    DispNum=np.floor(MaxIter/10)
    for i in range(MaxIter):
        # Line Search (Wolfe Condition)
        SearchVector=-H @ dp
        DescentVector, C0=LineSearch(x,y,SearchVector,dp,Net)
        s=DescentVector
        p=p+DescentVector
        Net.weight, Net.bias = VectorToMatrix(p,Net)
        dwNew, dbNew = AutomaticGradient(x,y,Net)
        dpNew = GradMatToVec(dwNew,dbNew,Net)
        yb=dpNew-dp
        dp=dpNew
        H=BFGS(s,yb,H)
        if np.mod(i,DispNum)==0:
            print("Iteration: {}, Cost: {:4.4f}".format(i+1, C0))
        CostList[0][i]=C0

    Net.OptimizationHistory=CostList
    error=y-Net.Evaluate(x)
    MAE=np.mean(abs(error))
    print("Max Iteration: {}, Mean Absolute Error: {:4.4f}".format(MaxIter, MAE))
    
    return Net
