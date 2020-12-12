import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PCA:
    raw_samples=()  ##原始样本空间
    n_samples=2000  ##样本个数
    raw_covariance=()  ##原始样本空间协方差矩阵
    reduced_samples=()   ##降维重构后样本空间
    raw_dimention=3   ##原始样本维度
    reduced_dimention=2   ##降维后样本的维度
    eigenvector=()  ##原始样本协方差矩阵的前M个（对应前M个特征值）特征向量
    def make_swiss_roll(self,noise=0.0, y_scale=100):   ##构造原始数据
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, self.n_samples))
        x = t * np.cos(t)
        y = y_scale * np.random.rand(1,self.n_samples)
        z = t * np.sin(t)
        self.raw_samples = np.concatenate((x, y, z))
        self.raw_samples += noise * np.random.randn(3, self.n_samples)  ##加入噪声
        self.raw_samples = np.mat(self.raw_samples.T)
    '''
        对矩阵X进行旋转变换
        theta为旋转的弧度
        axis为旋转的轴，合法值为'x','y'或'z'
    '''
    def rotate(self,theta=0,axis='x'):
        if axis == 'x':
            rotate = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
            return np.dot(rotate, self.raw_samples.T).T
        elif axis == 'y':
            rotate = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
            return np.dot(rotate, self.raw_samples.T).T
        elif axis == 'z':
            rotate = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            return np.dot(rotate, self.raw_samples.T).T
        else:
            print('错误的旋转轴')
            return self.raw_samples
def show_3D(raw_sample):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=80)
    ax.scatter(raw_sample[:, 0], raw_sample[:, 1], raw_sample[:, 2], c=raw_sample[:, 0], cmap=plt.cm.gnuplot())
    plt.show()
def show_2D(recon_data):
    plt.scatter(recon_data[:,0].tolist(), recon_data[:,1].tolist(), c=recon_data[:,0].tolist(), cmap=plt.cm.gnuplot())
    plt.show()
def cal_covariance(data):
    sample,feature=data.shape
    data_mean = np.sum(data, 0) / sample
    c_data = data - data_mean # 中心化
    covMat = np.dot(c_data.T, c_data)
    return covMat,c_data,data_mean
def cal_engin(covMat,reduced_dimention):
    eigenvalue, eigenvector = np.linalg.eig(covMat)
    index = np.argsort(eigenvalue)
    eigenvector = eigenvector[:,index[:-(reduced_dimention + 1):-1]]
    return np.real(eigenvector)
def cal_reduced_sample(c_data,eigenvector,data_mean):
    pca_data = np.dot(c_data, eigenvector)  # 计算降维后的数据
    print(f"shape of pca_data is {pca_data}")
    recon_data = np.dot(pca_data, eigenvector.T) + data_mean  # 重构数据
    return recon_data

size = (50, 50)
'''
    从file_path中读取面部图像数据
'''
def read_faces(file_path):
    file_list = os.listdir(file_path)
    data = []
    i = 1
    plt.figure(figsize=size)
    for file in file_list:
        path = os.path.join(file_path, file)##拼接路径
        plt.subplot(3, 4, i)
        with open(path) as f:
            img = cv2.imread(path) # 读取图像
            img = cv2.resize(img, size) # 压缩图像至size大小
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 三通道转换为灰度图
            plt.imshow(img_gray) # 预览
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w) # 对(h,w)的图像数据拉平
            data.append(img_col)
        i += 1
    plt.show()
    return np.array(data)
'''
    计算峰值信噪比psnr
'''
def psnr(img1, img2):
   mse = np.mean((img1 / 255. - img2 / 255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
if __name__ == '__main__':
    # pca=PCA()
    # pca.make_swiss_roll()
    # covMat,c_data,data_mean=cal_covariance(pca.raw_samples)
    # eigenvectors=cal_engin(covMat,2)
    # recon_data=cal_reduced_sample(c_data,eigenvectors,data_mean)
    # show_3D(pca.raw_samples)
    # show_2D(recon_data)
    data=read_faces("photos")
    n_samples,features=data.shape
    covMat,c_data,data_mean=cal_covariance(data)
    eigenvectors=cal_engin(covMat,20)
    recon_data=cal_reduced_sample(c_data,eigenvectors,data_mean)
    plt.figure(figsize=size)
    for i in range(n_samples):
        plt.subplot(3, 4, i + 1)
        plt.imshow(recon_data[i].reshape(size))
    plt.show()
    print("信噪比如下：")
    for i in range(n_samples):
        a = psnr(data[i], recon_data[i])
        print('图', i, '的信噪比: ', a)