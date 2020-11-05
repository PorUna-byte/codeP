import numpy as np
import math
import matplotlib.pyplot as plt
class Logistic:
    X=() #样本空间
    Y=() #样本空间中各样本的维度
    N=1000 #样本空间中样本点的个数
    m=2 #原始样本点的维数，扩充之后为m+1维，最后一维为1
    W=()#扩充之后的参数矩阵,m+1维,所得的线性分类器方程为:W[0][0]*x+W[1][0]*y+W[2][0]=0
    # sigmoid函数
    @staticmethod
    def sig(wx):
        if wx < -10:
            return 0
        else:
            return 1 / (1 + math.exp(-wx))
    @staticmethod
    def sig_plus(wx):
        if(wx<-10 or wx>10):
            return 0
        else:
            return 1 /(2+math.exp(wx)+math.exp(-wx))

    def cal_Hessian_nopunish(self):
        Hessian = np.zeros((self.m + 1, self.m + 1))
        for i in range(self.m + 1):
            for j in range(self.m + 1):
                for k in range(self.N):
                    Hessian[i][j] += self.X[k][i] * self.X[k][j] * self.sig_plus(np.dot(self.W.T, self.X[k]))
        return Hessian

    def cal_Hessian_withpunish(self, lamda):
        Hessian = np.zeros((self.m + 1, self.m + 1))
        for i in range(self.m + 1):
            for j in range(self.m + 1):
                for k in range(self.N):
                    Hessian[i][j] += self.X[k][i] * self.X[k][j] * self.sig_plus(np.dot(self.W.T, self.X[k]))
                Hessian[i][j] /= self.N
                if (i == j):
                    Hessian[i][j] += lamda
        return Hessian

    def calloss_nopunish(self):
        sum=0
        for i in range(self.N):
            WX=np.dot(self.W.T,self.X[i])
            sum+=-self.Y[i][0]*WX
            if WX > 0:
                sum += WX + math.log(1 + math.exp(-WX))
            else:
                sum += math.log(1 + math.exp(WX))
        return sum
    def calloss_withpunish(self,lamda):
        sum=self.cal_Hessian_withpunish(lamda)
        sum/=self.N
        sum+=1/(2*self.N)*lamda*np.dot(self.W.T,self.W)
        return sum
    def calloss_gradient_nopunish(self):
        delta=np.zeros((self.m+1,1))
        sum=0
        for j in range(self.m+1):
            for i in range(self.N):
                sum += self.X[i][j] * (-self.Y[i] + self.sig(np.dot(self.W.T, self.X[i])))
            delta[j][0]=sum
            sum=0
        print(f"delta={delta}")
        return delta
    def calloss_gradient_withpunish(self,lamda):
        delta=np.zeros((self.m+1,1))
        sum = 0
        for j in range(self.m+1):
            for i in range(self.N):
                sum +=(1/(float)(self.N))*(self.X[i][j] * (-self.Y[i] + self.sig(np.dot(self.W.T, self.X[i])))+lamda*self.W[j][0])
            delta[j][0] = sum
            sum = 0
        return delta
    def calW_nopunish(self,ita):
        self.W=np.zeros((self.m+1,1))
        delta = self.calloss_gradient_nopunish()
        while(np.dot(delta.T,delta)>2):
            for j in range(self.m + 1):
                self.W[j][0] = self.W[j][0] - ita * delta[j][0]
            delta = self.calloss_gradient_nopunish()
            print(f"W={self.W}")
    def calW_withpunish(self,ita,lamda):
        self.W = np.zeros((self.m + 1, 1))
        delta = self.calloss_gradient_withpunish(lamda)
        while (np.dot(delta.T, delta) > 2):
            for j in range(self.m + 1):
                self.W[j][0] = self.W[j][0] - ita * delta[j][0]
            delta = self.calloss_gradient_withpunish(lamda)

    def Newton_nopunish(self,step):
        count=0
        self.W = np.zeros((self.m + 1, 1))
        while count < step:
            g = self.calloss_gradient_nopunish()
            H = self.cal_Hessian_nopunish()
            self.W -= np.dot(np.linalg.inv(H), g)
            count+=1

    def Newton_withpunish(self,step,lamda):
        count = 0
        self.W = np.zeros((self.m + 1, 1))
        while count < step:
            g = self.calloss_gradient_withpunish(lamda)
            H = self.cal_Hessian_withpunish(lamda)
            self.W -= np.dot(np.linalg.inv(H), g)
            count += 1

    def cal_correct_rate(self,X,Y,size):
        cnt=0
        for i in range(size):
            WX=np.dot(self.W.T,X[i])
            if(WX>0 and Y[i][0]>0):
                cnt+=1
            if(WX<0 and Y[i][0]==0):
                cnt+=1
        return (float)(cnt)/size
    def read_train_set(self,filename):
        cnt=0
        with open(filename) as file_object:
            for line in file_object:
                ans=line.split(',')
                if ans[0]=='L' or ans[0]=='R':
                    cnt+=1
            self.N=cnt-50
            self.X=np.zeros((self.N,self.m+1))
            self.Y=np.zeros((self.N,1))
            X=np.zeros((50,self.m+1))
            Y=np.zeros((50,1))
        i=0
        with open(filename) as file_object:
            for line_ in file_object:
                ans_=line_.split(',')
                if ans_[0]=='L':
                    if i < cnt-50:
                       self.Y[i][0] = 0
                       self.X[i][0] = int(ans_[1])
                       self.X[i][1] = int(ans_[2])
                       self.X[i][2] = int(ans_[3])
                       self.X[i][3] = int(ans_[4][0])
                       self.X[i][4] = 1
                    else:
                       Y[i+50-cnt][0] = 0
                       X[i+50-cnt][0] = int(ans_[1])
                       X[i+50-cnt][1] = int(ans_[2])
                       X[i+50-cnt][2] = int(ans_[3])
                       X[i+50-cnt][3] = int(ans_[4][0])
                       X[i+50-cnt][4] = 1
                    i+=1
                if ans_[0]=='R':
                    if i < cnt-50:
                        self.Y[i][0] = 1
                        self.X[i][0] = int(ans_[1])
                        self.X[i][1] = int(ans_[2])
                        self.X[i][2] = int(ans_[3])
                        self.X[i][3] = int(ans_[4][0])
                        self.X[i][4] = 1
                    else:
                        Y[i+50-cnt][0] = 1
                        X[i+50-cnt][0] = int(ans_[1])
                        X[i+50-cnt][1] = int(ans_[2])
                        X[i+50-cnt][2] = int(ans_[3])
                        X[i+50-cnt][3] = int(ans_[4][0])
                        X[i+50-cnt][4] = 1
                    i+=1
        return (X,Y)

def initial(cov):
        logis = Logistic()
        mean0 = [2, 3]
        x0 = np.random.multivariate_normal(mean0, cov, 500).T
        mean1 = [7, 8]
        x1 = np.random.multivariate_normal(mean1, cov, 500).T
        logis.N = 1000
        logis.X = np.zeros((1000, 3))
        logis.Y = np.zeros((1000, 1))
        for i in range(500):
            logis.X[i][0] = x0[0][i]
            logis.X[i + 500][0] = x1[0][i]
            logis.X[i][1] = x0[1][i]
            logis.X[i + 500][1] = x1[1][i]
            logis.X[i][2] = 1
            logis.X[i + 500][2] = 1
            logis.Y[i][0] = 0
            logis.Y[i + 500][0] = 1
        return logis
if __name__ == '__main__':
    # cov = np.mat([[1, 0], [0, 1]])
    # logis=initial(cov)
    # mean0 = [2, 3]
    # x_0 = np.random.multivariate_normal(mean0, cov, 500).T
    # mean1 = [7, 8]
    # x_1 = np.random.multivariate_normal(mean1, cov, 500).T
    # X = np.zeros((1000, 3))
    # Y = np.zeros((1000, 1))
    # for i in range(500):
    #     X[i][0] = x_0[0][i]
    #     X[i + 500][0] = x_1[0][i]
    #     X[i][1] = x_0[1][i]
    #     X[i + 500][1] = x_1[1][i]
    #     X[i][2] = 1
    #     X[i + 500][2] = 1
    #     Y[i] = 0
    #     Y[i + 500] = 1
    logis=Logistic()
    logis.m=4
    (X,Y)=logis.read_train_set("balance-scale.data")
    logis.calW_nopunish(0.001)
    ans = "shot rate= " + str(logis.cal_correct_rate(logis.X,logis.Y,logis.N))
    print(ans)

