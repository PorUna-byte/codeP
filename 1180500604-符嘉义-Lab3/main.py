import numpy as np
import random
import math
import matplotlib.pyplot as plt
class GMM_EM_Kmeans:
    X=()  ##样本空间
    Z=()  ##潜变量
    U=()  ##聚类中心
    N=1500  ##数据量
    D=2     ##维度
    K=3     ##所聚的类数
    log_likelyhood=[] ##每一次迭代之后的对数可能性
    gamma=() ##posterior(样本x[n]由第k个高斯模型产生)
    Sigma=() ##各单高斯模型的协方差
    pi=() ##各单高斯模型所占的权重，也就是prior(样本x[n]由第k个高斯模型产生),其和为1
    def initial_k_means(self):
        cov0 = np.mat([[1, 0], [0, 1]])
        mean0 = [2, 3]
        x_0 = np.random.multivariate_normal(mean0, cov0, 500).T
        mean1 = [8, 2]
        cov1 = np.mat([[2,1],[1,2]])
        x_1 = np.random.multivariate_normal(mean1, cov1, 500).T
        mean2 = [7, 8]
        cov2=np.mat([[2,-1],[-1,2]])
        x_2 = np.random.multivariate_normal(mean2, cov2, 500).T
        self.X = np.zeros((self.N, self.D))
        self.Z = np.zeros((self.N, self.K))
        self.U = np.zeros((self.K,self.D))
        for i in range(self.K):
            for j in range(self.D):
                self.U[i][j]=random.random()*10 ##随机选择k个点作为k个单高斯模型的均值点,这里k=3
        for i in range(500):
            self.X[i][0] = x_0[0][i]
            self.X[i + 500][0] = x_1[0][i]
            self.X[i + 1000][0] = x_2[0][i]
            self.X[i][1] = x_0[1][i]
            self.X[i + 500][1] = x_1[1][i]
            self.X[i + 1000][1] = x_2[1][i]
    def K_Means(self,step):
         plt.figure(figsize=(48, 48))
         for t in  range(step):
            self.Z = np.zeros((self.N, self.K))
            for i in range(self.N): ##E-step
                min=0
                for j in range(self.K):
                    if(np.dot((self.X[i]-self.U[j]),(self.X[i]-self.U[j])).T<np.dot((self.X[i]-self.U[min]),(self.X[i]-self.U[min])).T):
                        min=j
                self.Z[i][min]=1
            for k in range(self.K):  ##M-step
                sum_zx=np.zeros((1,self.D))
                sum_z=0
                for i in range(self.N):
                    sum_zx+=self.Z[i][k]*self.X[i]
                    sum_z+=self.Z[i][k]
                self.U[k]=sum_zx[0]/sum_z
            if t==0:
                plt.subplot(2,2,1)
                self.plot_k_means()
            elif t==1:
                plt.subplot(2,2,2)
                self.plot_k_means()
            elif t==2:
                plt.subplot(2,2,3)
                self.plot_k_means()
            elif t==8:
                plt.subplot(2,2,4)
                self.plot_k_means()
         plt.show()
    def plot_k_means(self):
        first=0
        second=0
        third=0
        fc=0
        sc=0
        tc=0
        for i in range(self.N):
            if(self.Z[i][0]>0):
                first+=1
            elif(self.Z[i][1]>0):
                second+=1
            else:
                third+=1
        x1=np.zeros((first,2))
        x2=np.zeros((second,2))
        x3=np.zeros((third,2))
        for i in range(self.N):
            if(self.Z[i][0]>0):
                x1[fc]=self.X[i]
                fc+=1
            elif(self.Z[i][1]>0):
                x2[sc]=self.X[i]
                sc+=1
            else:
                x3[tc]=self.X[i]
                tc+=1
        for i in range (self.K):
            plt.scatter(self.U[i][0], self.U[i][1], marker='+', color='black', s=5000)
        plt.scatter(x1.T[0], x1.T[1], marker='.', color='green')
        plt.scatter(x2.T[0], x2.T[1], marker='.', color='blue')
        plt.scatter(x3.T[0],x3.T[1],marker='.', color='red')
    def initial_GMM_EM(self):
        self.pi=np.zeros((self.K,1))
        for k in range(self.K):
            self.pi[k][0]=1.0/self.K
        self.gamma=np.zeros((self.N,self.K))
        self.Sigma=np.zeros((self.K,self.D,self.D))
        self.U = np.zeros((self.K,self.D))
        self.Z = np.zeros((self.N, self.K))
        for i in range(self.K):
            for j in range(self.D):
                self.U[i][j]=random.random()*10 ##随机选择K个点作为K个单高斯模型的均值点
        print(self.U)
        flag=True
        while(flag):  ##此循环保证每个样本中心都有点分配到，此步骤可能耗时较长，在一些奇葩的数据上。
            for i in range(self.N):  ##E-step
                min = 0
                for j in range(self.K):
                    if (np.dot((self.X[i] - self.U[j]), (self.X[i] - self.U[j])).T < np.dot((self.X[i] - self.U[min]), (
                            self.X[i] - self.U[min])).T):
                        min = j
                self.Z[i][min] = 1
            flag=False
            for k in range(self.K):  ##M-step
                sum_zx = np.zeros((1, self.D))
                sum_z = 0
                for i in range(self.N):
                    sum_zx += self.Z[i][k] * self.X[i]
                    sum_z += self.Z[i][k]
                if(sum_z==0):
                   print(f'第{k}个样本中心无点分配')
                   print(f'重新初始化为:')
                   for d in range(self.D):
                       self.U[k][d]=random.random()*5
                   print(f'self.U[{k}]={self.U[k]}')
                   flag=True
                else:
                   self.U[k] = sum_zx[0] / sum_z
            for k in range(self.K):
                self.Sigma[k]=np.identity(self.D)
    def GMM_EM(self,step):
        for t in range(step):    ##迭代step步
            for n in range(self.N):  ##E-step
                for k in range(self.K):
                    sum=0
                    for j in range(self.K):
                        sum+=self.pi[j][0]*caln_x_u_sig(self.X[n],self.U[j],self.Sigma[j],self.D)
                    self.gamma[n][k]=self.pi[k][0]*caln_x_u_sig(self.X[n],self.U[k],self.Sigma[k],self.D)/sum
            for k in range(self.K):   ##M-step
                N_k=0
                for n in range(self.N):
                    N_k+=self.gamma[n][k]
                for d in range(self.D):
                    self.U[k][d]=0
                for n in range(self.N):
                    self.U[k]+=self.gamma[n][k]*self.X[n]
                self.U[k]/=N_k
                self.Sigma[k]=np.zeros((self.D,self.D))
                for n in range(self.N):
                    self.Sigma[k]+=self.gamma[n][k]*np.outer((self.X[n]-self.U[k]).T,(self.X[n]-self.U[k]))
                self.Sigma[k]/=N_k
                self.pi[k][0]=N_k/self.N
            self.callog_likelyhood()
    def plot_GMM(self):
        first=0
        second=0
        third=0
        fc=0
        sc=0
        tc=0
        for i in range(self.N):
            if(self.gamma[i][0]>self.gamma[i][1] and self.gamma[i][0]>self.gamma[i][2]):
                first+=1
            elif(self.gamma[i][1] > self.gamma[i][0] and self.gamma[i][1] > self.gamma[i][2]):
                second+=1
            else:
                third+=1
        x1=np.zeros((first,self.D))
        x2=np.zeros((second,self.D))
        x3=np.zeros((third,self.D))
        for i in range(self.N):
            if(self.gamma[i][0]>self.gamma[i][1] and self.gamma[i][0]>self.gamma[i][2]):
                x1[fc]=self.X[i]
                fc+=1
            elif(self.gamma[i][1]>self.gamma[i][0] and self.gamma[i][1]>self.gamma[i][2]):
                x2[sc]=self.X[i]
                sc+=1
            else:
                x3[tc]=self.X[i]
                tc+=1
        for i in range (self.K):
            plt.scatter(self.U[i][0], self.U[i][1], marker='+', color='black', s=5000)
        plt.scatter(x1.T[0], x1.T[1], marker='.', color='green')
        plt.scatter(x2.T[0], x2.T[1], marker='.', color='blue')
        plt.scatter(x3.T[0],x3.T[1],marker='.', color='red')
    def callog_likelyhood(self):
        ans=0
        for n in range(self.N):
            sum=0
            for k in range(self.K):
                sum+=self.pi[k][0]*caln_x_u_sig(self.X[n],self.U[k],self.Sigma[k],self.D)
            ans+=math.log(sum)
        self.log_likelyhood.append(ans)
    def plot_likelyhood(self,step):
        st=[]
        for i in range(step):
            st.append(i)
        plt.scatter(st,self.log_likelyhood,marker='.',color='red',s=300)
        plt.plot(st,self.log_likelyhood,color='blue')
    def read_data(self,filename):
        dict={'vhigh':4,'high':3,'med':2,'low':1,'2':2,'3':3,'4':4,'5more':5,'more':5,'small':1,'big':3}
        cnt=0
        with open(filename) as file_obj:
            for line in file_obj:
                cnt+=1
        self.N=cnt
        self.D=6
        self.K=4
        self.X=np.zeros((self.N,self.D))
        self.pi = np.zeros((self.K, 1))
        for k in range(self.K):
            self.pi[k][0] = 1.0 / self.K
        self.gamma = np.zeros((self.N, self.K))
        self.Sigma = np.zeros((self.K, self.D, self.D))
        self.U = np.zeros((self.K, self.D))
        self.Z = np.zeros((self.N, self.K))
        for k in range(self.K):
            self.Sigma[k] = np.identity(self.D)
        i=0
        with open(filename) as file_obj:
            for line_ in file_obj:
                ans=line_.split(',')
                for j in range(6):
                    self.X[i][j] = dict[ans[j]]
                i+=1
        self.U[0]=self.X[self.N-1]
        self.U[1]=self.X[1294]
        self.U[2]=self.X[854]
        self.U[3]=self.X[234]
        flag=True
        while(flag):  ##此循环保证每个样本中心都有点分配到，此步骤可能耗时较长，在一些奇葩的数据上。
            for i in range(self.N):  ##E-step
                min = 0
                for j in range(self.K):
                    if (np.dot((self.X[i] - self.U[j]), (self.X[i] - self.U[j])).T < np.dot((self.X[i] - self.U[min]), (
                            self.X[i] - self.U[min])).T):
                        min = j
                self.Z[i][min] = 1
            flag=False
            for k in range(self.K):  ##M-step
                sum_zx = np.zeros((1, self.D))
                sum_z = 0
                for i in range(self.N):
                    sum_zx += self.Z[i][k] * self.X[i]
                    sum_z += self.Z[i][k]
                if(sum_z==0):
                   print(f'第{k}个样本中心无点分配')
                   print(f'重新初始化为:')
                   for d in range(self.D):
                       self.U[k][d]=random.random()*5
                   print(f'self.U[{k}]={self.U[k]}')
                   flag=True
                else:
                   self.U[k] = sum_zx[0] / sum_z
def caln_x_u_sig(x,u,sigma,d):
    if((2*math.pi**(d/2)*np.linalg.det(sigma)**0.5)==0):
        print('error 0!')
    coff=1/(2*math.pi**(d/2)*np.linalg.det(sigma)**0.5)
    para=math.exp(-0.5*np.dot(np.dot(x-u,np.linalg.inv(sigma)),(x-u).T))
    return coff*para



if __name__ == '__main__':
    # gmm.initial_k_means()
    # gmm.K_Means(10)
    # gmm=GMM_EM_Kmeans()
    # gmm.initial_k_means()
    # gmm.initial_GMM_EM()
    # gmm.GMM_EM(20)
    # plt.figure(figsize=(48, 48))
    # plt.subplot(2,2,1)
    # gmm.plot_likelyhood(20)
    # plt.subplot(2,2,2)
    # gmm.plot_GMM()
    # plt.show()
    # print(f'U is :{gmm.U}')
    # print(f'Sigma is\n:{gmm.Sigma}')
    # print(f'pi is :{gmm.pi}')
    gmm=GMM_EM_Kmeans()
    gmm.read_data('car.data')
    gmm.GMM_EM(7)
    print(f'U is :{gmm.U}')
    print(f'Sigma is:\n{gmm.Sigma}')
    print(f'pi is :{gmm.pi}')
    gmm.plot_likelyhood(7)
    plt.show()


