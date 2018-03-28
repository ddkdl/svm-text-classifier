# Author Alvaro Esperanca

import numpy as np
import cvxopt.solvers
from Kernel import Kernel
from LinearKernel import LinearKernel

class SVM(object):
    def __init__(self, kernel=Kernel, C=None):
        self.kernel = kernel
        self.C = C
        self.b = None
        self.w = None
        self.a = None
        self.sv = None
        self.sv_y = None
        if self.C is not None: self.C = float(self.C)
        
    
    def saveModel(self):
        with open("linearModel.svm", "wb") as outputFile:
            outputFile.write("%s\n" % (self.C or "None"))
            
            if self.w is None:
                outputFile.write("%s\n" % ("None"))
            else:
                for item in self.w:
                    outputFile.write("%s " % (item))
                outputFile.write("\n")
            
            
            outputFile.write("%s\n" % (self.b or "None"))
            
            if self.a is None:
                outputFile.write("%s\n" % ("None"))
            else:
                for item in self.a:
                    outputFile.write("%s " % (item))
                outputFile.write("\n")
                
                
            if self.sv is None:
                outputFile.write("%s\n" % ("None"))
            else:
                outputFile.write("%s\n" % len(self.sv) )
                for item in self.sv:               
                    for element in item:
                        outputFile.write("%s " % (element))
                    outputFile.write("\n")
                
                
            if self.sv_y is None:
                outputFile.write("%s\n" % ("None"))
            else:
                for item in self.sv_y:
                    outputFile.write("%s " % (item))
                outputFile.write("\n")
                
    def loadModel(self):
        with open("linearModel.svm", "rb") as inputFile:
            line = inputFile.readline()
            if line == "None":
                self.C = None
            else:
                self.C = float(line)
                
            self.w = []
            line = inputFile.readline()
            if line == "None":
                self.w = None
            else:
                for item in line.split():
                    self.w.append(float(item))
                    
                    
            
            line = inputFile.readline()
            if line == "None":
                self.b = None
            else:
                self.b = float(line)
                
            self.a = []
            line = inputFile.readline()
            if line == "None":
                self.a = None
            else:
                for item in line.split():
                    self.a.append(float(item))
                    
                    
            self.sv = []
            temp = []
            
            line = inputFile.readline()
            if line == "None":
                self.sv = None
            else:
                for i in range(int(line)):
                    otherLine = inputFile.readline()
                    for item in otherLine.split():
                        temp.append(float(item))
                    self.sv.append(temp)
                    temp[:] = []
                    
                    
            self.sv_y = []
            line = inputFile.readline()
            if line == "None":
                self.sv_y = None
            else:
                for item in line.split():
                    self.sv_y.append(float(item))
            
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel.eval(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)

        q = cvxopt.matrix(np.ones(n_samples) * -1)

        A = cvxopt.matrix(y, (1, n_samples), 'd')

        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        if type( self.kernel ) is LinearKernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel.eval(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
