# RANSAC算法是能够排除异常数据干扰的一个回归算法
import numpy as np
import random
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self,x,y):
        x = process_features(x)
        self.x = np.array(x)
        self.y = np.array(y)

    def fit(self,X,y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        pass
    def predict(self,X):
        # X = process_features(X)
        return X.dot(self.w)

    def predict_result(self,X):
        X = process_features(X)
        return X.dot(self.w)
    # RANSAC算法
    def RANSAC(self,N,d,k):
        m,n = self.x.shape
        w_list = []
        r_list = []
        t = 0
        while t<=N:
            # 随机取子集
            a,b = random.sample(range(m),2)
            if a>b:
                a,b = b,a
            # 子集数量大于k，即要求的取点数量
            if b-a>=k:
                # 根据子集建模
                self.fit(self.x[a:b,:],self.y[a:b])
                y_true = self.y
                y_pred = self.predict(self.x)
                # 计算训练数据，回归数据的误差
                B = abs(y_true-y_pred)
                Bt = []
                By = []

                # 选择出误差小于d的数据
                for i in range(len(B)):
                    if B[i]<d:
                        Bt.append(self.x[i,:])
                        By.append(self.y[i])
                Bt = np.array(Bt)
                By = np.array(By)

                # 若误差小于d的数据数量大于k，则使用这些数据建立新的墨香
                if len(Bt)>k:
                    self.fit(Bt,By)
                    y_pred =self.predict(Bt)
                    y_true = By
                # 计算均方误差，并保存本次模型与均方误差
                r =mean_squared_error(y_true,y_pred)
                w_list.append(self.w)
                r_list.append(r)
                t = t+1
                pass
            pass
        index_min = np.argmin(r_list)
        self.w = w_list[index_min]
        pass

# 均方误差
# np.average() 用于求平均
def mean_squared_error(y_true,y_pred):
    return np.average((y_true-y_pred)**2,axis=0)


# 生成异常数据
def generate_samples(m,k):
    X_normal = 2 * (np.random.rand(m,1)-0.5)
    y_normal = X_normal+np.random.normal(0,0.1,(m,1))
    X_outlier = 2 * (np.random.rand(k,1)-0.5)
    y_outlier = X_outlier+np.random.normal(3,0.1,(k,1))
    X = np.concatenate((X_normal,X_outlier),axis=0)
    y = np.concatenate((y_normal,y_outlier),axis= 0)
    return X, y

def process_features(X):
    m,n = X.shape
    X = np.c_[np.ones((m,1)),X]
    return X


# np.random.seed(0)
# X_original,y = generate_samples(100,5)
# print(X_original.shape)
# X = process_features(X_original)
# # print(X_original,'\n',X)
# model =LinearRegression(X_original,y)
# # model.fit(X,y)
# # y_pred = model.predict(X)
# model.RANSAC(5,3,3)
# y_RANSAC = model.predict_result(X_original)
# print(mean_squared_error(y,y_RANSAC))
# plt.scatter(X_original,y)
# plt.scatter(X_original,y_RANSAC)
# # plt.plot(X_original,y_pred,color = "yellow")
# plt.plot(X_original,y_RANSAC,color = "red")
# plt.show()


per = np.load("ori.npy")
ran = np.load("ran.npy")
pre_x = [i+1 for i in range(len(per))]
ran_x = [i+1 for i in range(len(ran))]
plt.scatter(pre_x,per,color = "yellow")
plt.plot(ran_x,ran,color = "black")
#
plt.show()