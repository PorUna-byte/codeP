import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
def changeImage(path):
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 为了避免溢出，对数据进行一个缩放，缩小100倍
    data = np.array(data).reshape(height,width)/100
    return data
def pca_pic(data,k):
    sample,featureNum = data.shape
    mean = np.array([np.mean(data[:,i]) for i in range(featureNum)]) # 求均值
    normal_data = data - mean  # 去中心化
    # 得到协方差矩阵
    matri = np.dot(np.transpose(normal_data),normal_data)
    val,vec = np.linalg.eig(matri)
    # 求前k个向量
    index = np.argsort(val)
    vecIndex = index[:-(k+1):-1]
    feature = vec[:,vecIndex]
    #降维后的数据
    new_data = np.dot(normal_data,feature)
    # 图片显示需要还原到原空间而不是直接返回降维后的数据
    # 将降维后的数据映射回原空间
    print(f'new_data.shape is {new_data.shape}')
    rec_data = np.dot(new_data,np.transpose(feature))+ mean
    print(f'rec_data.shape is {rec_data.shape}')
    return rec_data
def error(data,recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata
    # 计算两幅图像之间的误差率，即信息丢失率
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2/sum1
    return error
if __name__ == '__main__':
    x = changeImage('0.jpg')
    print(x.shape)
    plt.imshow(x, cmap=plt.cm.gray) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    plt.subplot(2, 2, 1)
    y = pca_pic(x, 1)
    plt.imshow(y.real, cmap=plt.cm.gray)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.subplot(2, 2, 2)
    print(error(x, y))
    y = pca_pic(x, 5)
    plt.imshow(y.real, cmap=plt.cm.gray)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.subplot(2, 2, 3)
    print(error(x, y))
    y = pca_pic(x, 10)
    plt.imshow(y.real, cmap=plt.cm.gray)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.subplot(2, 2, 4)
    print(error(x, y))
    y = pca_pic(x, 20)
    plt.imshow(y.real, cmap=plt.cm.gray)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    print(error(x, y))
    plt.show()