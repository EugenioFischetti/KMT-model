import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from scipy.integrate import quad #Single integral
from scipy.integrate import dblquad
from scipy.integrate import tplquad # Integral tripla
from PIL import Image

def KMT(K, M, T, b1, n1, forma2, escala2, l_tx, b, ci, cb, cr, cf, c, mi):
    ######### Definitions #######################################################
    def fx(x):
        return (b1/n1**b1)*(x**(b1-1))*np.exp(-(x/n1)**b1)
    
    def Fx(x):  # Weibull CDF (minor defect)
        return quad(fx, 0, x)[0]
    
    def Rx(x):  # Weibull Reliability
        return 1 - Fx(x)
    
    def fy(y):  # Weibull density (major defect)
        return (forma2/escala2)*((y/escala2)**(forma2-1))*np.exp(-(y/escala2)**forma2)
    
    def Ry(y):  # Major defect reliability
        return quad(fy, y, np.inf)[0]
    
    def fh(h):  # Exponential delay time density
        return l_tx*np.exp(-l_tx*h)
    
    def Fh(h):  # Exponential CDF
        return 1 - np.exp(-l_tx*h)
    
    def Rh(h):  # Delay time reliability
        return np.exp(-l_tx*h)
    #####Scenarios#############################################################
    # CENÁRIO 1 - Defeito menor por degradação, defeito maior e falha chegam entre inspeções menores
    
    def P1li(l, i, T):
        def FP1(h, y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
        return (tplquad(FP1, (l*M+(i-1))*T, (l*M+i)*T, lambda x: 0, lambda x: (l*M+i)*T-x, lambda x, y: 0, lambda x, y: (l*M+i)*T-(x+y))[0])
    
    def P1(K, M, T):
        resultado_P1 = 0
        for l in range(0, K):
          for i in range(1,M+1):
                resultado_P1 = resultado_P1 + P1li(l, i, T)
        return resultado_P1
    
    def EC1(K, T, M):
        resultado_EC1 = 0
        def FC1(h, y, x):
            return (c*h)*np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
        for l in range(0, K):
            for i in range(1,M+1):
                  resultado_EC1 = resultado_EC1 + (((l * cb) + ((l*(M-1)+(i-1)) * ci) + cf) * P1li(l, i, T)) +  (tplquad(FC1, (l*M+(i-1))*T, (l*M+i)*T, lambda x: 0, lambda x: (l*M+i)*T-x, lambda x, y: 0, lambda x, y: (l*M+i)*T-(x+y))[0])
        return resultado_EC1
    
    def EL1(K, T, M):
        resultado_EL1 = 0
        for l in range(0, K):
            for i in range(1,M+1):
                  def FL1(h, y, x):
                    return (x+h+y) * np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
                  resultado_EL1 = resultado_EL1 +  (tplquad(FL1, (l*M+(i-1))*T, (l*M+i)*T, lambda x: 0, lambda x: (l*M+i)*T-x, lambda x, y: 0, lambda x, y: (l*M+i)*T-(x+y))[0])
        return resultado_EL1
    
    # CENÁRIO 2 - Defeito menor por choque, defeito maior e falha chegam entre inspeções menores
    
    def P2li(l, i, T):
        def FP2(h, y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
        return (tplquad(FP2, (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z, lambda z, y: 0, lambda z, y: (l*M+i)*T-(z+y))[0])
    
    def P2(K, M, T):
        resultado_P2 = 0
        for l in range(0, K):
          for i in range(1,M+1):
                resultado_P2 = resultado_P2 + P2li(l, i, T)
        return resultado_P2
    
    def EC2(K, T, M):
        resultado_EC2 = 0
        def FC2(h, y, z):
            return (c*h)*mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
        for l in range(0, K):
            for i in range(1,M+1):
                    resultado_EC2 = resultado_EC2 + (((l * cb) + ((l*(M-1)+(i-1)) * ci) + cf) * P2li(l, i, T)) +  (tplquad(FC2, (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z, lambda z, y: 0, lambda z, y: (l*M+i)*T-(z+y))[0])
        return resultado_EC2
    
    def EL2(K, T, M):
        resultado_EL2 = 0
        for l in range(0, K):
            for i in range(1,M+1):
                  def FL2(h, y, z):
                    return (z+h+y) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
                  resultado_EL2 = resultado_EL2 +  (tplquad(FL2, (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z, lambda z, y: 0, lambda z, y: (l*M+i)*T-(z+y))[0])
        return resultado_EL2
    
    
    # CENÁRIO 3 - Defeito menor por degradação, defeito maior e falha chegam entre inspeções menor após um ou mais falso negativo
    
    def P3lij(l, i, j, T):
        def FP3(h, y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
        return (b**(j-i)) * (tplquad(FP3, (l*M+(i-1))*T, (l*M+i)*T, lambda x: (l*M+j-1)*T-x, lambda x: (l*M+j)*T-x, lambda x, y: 0, lambda x, y: (l*M+j)*T-(x+y))[0])
    
    def P3(K, M, T):
        resultado_P3 = 0
        for l in range(0, K):
          for j in range(2,M+1):
            for i in range(1, j):
                resultado_P3 = resultado_P3 + P3lij(l, i, j, T)
        return resultado_P3
    
    def EC3(K, T, M):
        resultado_EC3 = 0
        def FC3(h, y, x):
            return (c*h)*np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
        for l in range(0, K):
            for j in range(2,M+1):
                for i in range(1,j):
                  resultado_EC3 = resultado_EC3 + (((l * cb) + ((l*(M-1)+(j-1)) * ci) + cf) * P3lij(l, i, j, T)) + (b**(j-i)) * (tplquad(FC3, (l*M+(i-1))*T, (l*M+i)*T, lambda x: (l*M+j-1)*T-x, lambda x: (l*M+j)*T-x, lambda x, y: 0, lambda x, y: (l*M+j)*T-(x+y))[0])
        return resultado_EC3
    
    def EL3(K, T, M):
        resultado_EL3 = 0
        for l in range(0, K):
            for j in range(2,M+1):
                for i in range(1,j):
                  def FL3(h, y, x):
                    return (x+h+y) * np.exp(-mi*x) * fx(x) * fy(y) * fh(h)
                  resultado_EL3 = resultado_EL3 + ((b**(j-i))* tplquad(FL3, (l*M+(i-1))*T, (l*M+i)*T, lambda x: (l*M+j-1)*T-x, lambda x: (l*M+j)*T-x, lambda x, y: 0, lambda x, y: (l*M+j)*T-(x+y))[0])
        return resultado_EL3
    
    
    # CENÁRIO 4 - Defeito menor por choque, defeito maior e falha chegam entre inspeções menor após um ou mais falso negativo
    
    def P4lij(l, i, j, T):
        def FP4(h, y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
        return (b**(j-i)) * (tplquad(FP4, (l*M+(i-1))*T, (l*M+i)*T, lambda z: (l*M+j-1)*T-z, lambda z: (l*M+j)*T-z, lambda z, y: 0, lambda z, y: (l*M+j)*T-(z+y))[0])
    
    def P4(K, M, T):
        resultado_P4 = 0
        for l in range(0, K):
          for j in range(2,M+1):
            for i in range(1, j):
                resultado_P4 = resultado_P4 + P4lij(l, i, j, T)
        return resultado_P4
    
    def EC4(K, T, M):
        resultado_EC4 = 0
        def FC4(h, y, z):
            return (c*h)*mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
        for l in range(0, K):
            for j in range(2,M+1):
                for i in range(1,j):
                    resultado_EC4 = resultado_EC4 + (((l * cb) + ((l*(M-1)+(j-1)) * ci) + cf) * P4lij(l, i, j, T)) + (b**(j-i)) * (tplquad(FC4, (l*M+(i-1))*T, (l*M+i)*T, lambda z: (l*M+j-1)*T-z, lambda z: (l*M+j)*T-z, lambda z, y: 0, lambda z, y: (l*M+j)*T-(z+y))[0])
        return resultado_EC4
    
    def EL4(K, T, M):
        resultado_EL4 = 0
        for l in range(0, K):
            for j in range(2,M+1):
                for i in range(1,j):
                  def FL4(h, y, z):
                    return (z+h+y) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * fh(h)
                  resultado_EL4 = resultado_EL4 + ((b**(j-i)) * (tplquad(FL4, (l*M+(i-1))*T, (l*M+i)*T, lambda z: (l*M+j-1)*T-z, lambda z: (l*M+j)*T-z, lambda z, y: 0, lambda z, y: (l*M+j)*T-(z+y))[0]))
        return resultado_EL4
    
    # CENÁRIO 5 – Defeito menor chega por degradação, defeito menor chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    
    def P5li(l, i, T):
        def FP5(y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * Rh((l*M+i)*T-(x+y))
        return (dblquad(FP5, (l*M+(i-1))*T, (l*M+i)*T, lambda x: 0, lambda x: (l*M+i)*T-x)[0])
    
    def P5(K, M, T):
        resultado_P5 = 0
        for l in range(0, K):
                for i in range(1,M):
                  resultado_P5 = resultado_P5 + P5li(l, i, T)
        return resultado_P5
    
    def EC5(K, T, M):
        resultado_EC5 = 0
        def FC5(y, x):
            return (c*((l*M+i)*T-x-y))*np.exp(-mi*x) * fx(x) * fy(y) * Rh((l*M+i)*T-(x+y))
        for l in range(0, K):
          for i in range(1,M):
            resultado_EC5 = resultado_EC5 + (((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P5li(l, i, T)) + (dblquad(FC5, (l*M+(i-1))*T, (l*M+i)*T, lambda x: 0, lambda x: (l*M+i)*T-x)[0])
        return resultado_EC5
    
    def EL5(K, T, M):
        resultado_EL5 = 0
        for l in range(0, K):
            for i in range(1,M):
                  resultado_EL5 = resultado_EL5 + ((((l*(M))+i) * T) * P5li(l, i, T))
                  #print("Para l =",l, "i=", i," Temos resultado_EL5:",resultado_EL5, "L:",((((l*(M-1))+i) * T)), "Prob:", P5li(l, i, T))
        return resultado_EL5
    
    # CENÁRIO 6 - Defeito menor chega por choque, defeito menor chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor
    
    def P6li(l, i, T):
        def FP6(y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l*M+i)*T-(z+y))
        return (dblquad(FP6, (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z)[0])
    
    def P6(K, M, T):
        resultado_P6 = 0
        for l in range(0, K):
                for i in range(1,M):
                  resultado_P6 = resultado_P6 + P6li(l, i, T)
        return resultado_P6
    
    def EC6(K, T, M):
        resultado_EC6 = 0
        def FC6(y, z):
            return (c*((l*M+i)*T-z-y))*mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l*M+i)*T-(z+y))
        for l in range(0, K):
          for i in range(1,M):
            resultado_EC6 = resultado_EC6 + (((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P6li(l, i, T)) + (dblquad(FC6, (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z)[0])
        return resultado_EC6
    
    def EL6(K, T, M):
        resultado_EL6 = 0
        for l in range(0, K):
          for i in range(1,M):
            resultado_EL6 = resultado_EL6 + ((((l*(M))+i) * T) * P6li(l, i, T))
            #print("l:",l,"i:",i, "Resultado:",resultado_EL6, "L:",((((l*(M-1))+i) * T)), "Prob:", P6li(l, i, T))
        return resultado_EL6
    
    # CENÁRIO 7 – Defeito menor chega por degradação, defeito menor chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    
    def P7lij(l, i, j, T):
        def FP7(y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * Rh((l*M+j)*T-(x+y))
        return (b**(j-i)) * (dblquad(FP7, (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
    
    def P7(K, M, T):
        resultado_P7 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_P7 = resultado_P7 + P7lij(l, i, j, T)
        return resultado_P7
    
    def EC7(K, T, M):
        resultado_EC7 = 0
        def FC7(y, x):
            return (c*((l*M+j)*T-x-y))*np.exp(-mi*x) * fx(x) * fy(y) * Rh((l*M+j)*T-(x+y))
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EC7 = resultado_EC7 + (((l * cb) + (((l*(M-1))+j) * ci) + cr) * P7lij(l, i, j, T)) + (b**(j-i)) * (dblquad(FC7, (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
    
        return resultado_EC7
    
    def EL7(K, T, M):
        resultado_EL7 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EL7 = resultado_EL7 + ((((l*(M))+j) * T) * P7lij(l, i, j, T))
                  #print("l:",l,"j",j,"i:",i, "Resultado:",resultado_EL7, "L:",((((l*(M-1))+j) * T)), "Prob:", P7lij(l, i, j, T))
        return resultado_EL7
    
    # CENÁRIO 8 - Defeito menor chega por choque, defeito menor chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor
    
    def P8lij(l, i, j, T):
        def FP8(y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l*M+j)*T-(z+y))
        return (b**(j-i)) * (dblquad(FP8, (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
    
    def P8(K, M, T):
        resultado_P8 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_P8 = resultado_P8 + P8lij(l, i, j, T)
        return resultado_P8
    
    def EC8(K, T, M):
        resultado_EC8 = 0
        def FC8(y, z):
            return (c*((l*M+j)*T-z-y))*mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l*M+j)*T-(z+y))
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EC8 = resultado_EC8 + (((l * cb) + (((l*(M-1))+j) * ci) + cr) * P8lij(l, i, j, T)) + (b**(j-i)) * (dblquad(FC8, (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
        return resultado_EC8
    
    def EL8(K, T, M):
        resultado_EL8 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EL8 = resultado_EL8 + ((((l*(M))+j) * T) * P8lij(l, i, j, T))
        return resultado_EL8
    
    # CENÁRIO 9 - Defeito menor chega por degradação e é substituído em inspeção menor
    
    def P9li(l, i, T):
        def FP9(x):
            return np.exp(-mi*x) * fx(x) * Ry((l*M+i)*T-x)
        return (1-b) * (quad(FP9, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P9(K, M, T):
        resultado_P9 = 0
        for l in range(0, K):
                for i in range(1,M):
                  resultado_P9 = resultado_P9 + P9li(l, i, T)
        return resultado_P9
    
    def EC9(K, T, M):
        resultado_EC9 = 0
        for l in range(0, K):
          for i in range(1,M):
            resultado_EC9= resultado_EC9 + (((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P9li(l, i, T))
        return resultado_EC9
    
    def EL9(K, T, M):
        resultado_EL9 = 0
        for l in range(0, K):
          for i in range(1,M):
            resultado_EL9 = resultado_EL9 + ((((l*(M))+i) * T) * P9li(l, i, T))
        return resultado_EL9
    
    
    # CENÁRIO 10 - Defeito menor chega por choque e é substituído em inspeção menor
    
    def P10li(l, i, T):
        def FP10(z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * Ry((l*M+i)*T-z)
        return (1-b) * (quad(FP10, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P10(K, M, T):
        resultado_P10 = 0
        for l in range(0, K):
                for i in range(1,M):
                  resultado_P10 = resultado_P10 + P10li(l, i, T)
        return resultado_P10
    
    def EC10(K, T, M):
        resultado_EC10 = 0
        for l in range(0, K):
          for i in range(1,M):
            resultado_EC10 = resultado_EC10 + (((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P10li(l, i, T))
        return resultado_EC10
    
    def EL10(K, T, M):
        resultado_EL10 = 0
        for l in range(0, K):
          for i in range(1,M):
            resultado_EL10 = resultado_EL10 + ((((l*(M))+i) * T) * P10li(l, i, T))
        return resultado_EL10
    
    
    # CENÁRIO 11 - Defeito menor chega por degradação e é substituído em inspeção menor
    
    def P11lij(l, i, j, T):
        def FP11(x):
            return np.exp(-mi*x) * fx(x) * Ry((l*M+j)*T-x)
        return (b**(j-i))* (1-b) * (quad(FP11, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P11(K, M, T):
        resultado_P11 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_P11 = resultado_P11 + P11lij(l, i, j, T)
        return resultado_P11
    
    def EC11(K, T, M):
        resultado_EC11 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EC11 = resultado_EC11 + (((l * cb) + (((l*(M-1))+j) * ci) + cr) * P11lij(l, i, j, T))
        return resultado_EC11
    
    def EL11(K, T, M):
        resultado_EL11 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EL11 = resultado_EL11 + ((((l*(M))+j) * T) * P11lij(l, i, j, T))
        return resultado_EL11
    
    
    # CENÁRIO 12 - Defeito menor chega por choque e é substituído em inspeção menor
    
    def P12lij(l, i, j, T):
        def FP12(z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * Ry((l*M+j)*T-z)
        return (b**(j-i))* (1-b) * (quad(FP12, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P12(K, M, T):
        resultado_P12 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_P12 = resultado_P12 + P12lij(l, i, j, T)
        return resultado_P12
    
    
    def EC12(K, T, M):
        resultado_EC12 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EC12 = resultado_EC12 + (((l * cb) + (((l*(M-1))+j) * ci) + cr) * P12lij(l, i, j, T))
        return resultado_EC12
    
    def EL12(K, T, M):
        resultado_EL12 = 0
        for l in range(0, K):
            for j in range(2,M):
                for i in range(1,j):
                  resultado_EL12 = resultado_EL12 + ((((l*(M))+j) * T) * P12lij(l, i, j, T))
        return resultado_EL12
    
    
    # CENÁRIO 13 - Defeito menor chega por degradação, entre inspeções menores, e é substituído em inspeção maior
    
    def P13li(l, i, T):
        def FP13(x):
            return np.exp(-mi*x) * fx(x) * Ry((l+1)*M*T-x)
        return (b**(M-i)) * (quad(FP13, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P13(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P13 = 0
            for l in range(0, K-1):
                for i in range(1,M+1):
                    resultado_P13 = resultado_P13 + P13li(l, i, T)
            return resultado_P13
    
    def EC13(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC13 = 0
            for l in range(0, K-1):
              for i in range(1,M+1):
                resultado_EC13 = resultado_EC13 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P13li(l, i, T))
            return resultado_EC13
    
    def EL13(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL13 = 0
            for l in range(0, K-1):
              for i in range(1,M+1):
                resultado_EL13 = resultado_EL13 + (((l+1)*M * T) * P13li(l, i, T))
            return resultado_EL13
    
    
    # CENÁRIO 14 - Defeito menor chega por choque e é substituído em inspeção maior
    
    def P14li(l, i, T):
        def FP14(z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * Ry((l+1)*M*T-z)
        return (b**(M-i)) *  (quad(FP14, (l*M+(i-1))*T, (l*M+i)*T)[0])
    
    def P14(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P14 = 0
            for l in range(0, K-1):
                for i in range(1,M+1):
                    resultado_P14 = resultado_P14 + P14li(l, i, T)
            return resultado_P14
    
    def EC14(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC14 = 0
            for l in range(0, K-1):
              for i in range(1,M+1):
                resultado_EC14 = resultado_EC14 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P14li(l, i, T))
            return resultado_EC14
    
    def EL14(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL14 = 0
            for l in range(0, K-1):
              for i in range(1,M+1):
                resultado_EL14 = resultado_EL14 + (((l+1)*M * T) * P14li(l, i, T))
            return resultado_EL14
    
    # CENÁRIO 15 - Defeito menor por degradação, defeito maior e substituição em inspeção maior.
    
    def P15l(l, M, T):
        def FP15(y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * Rh((l+1)*M*T-(x+y))
        return (dblquad(FP15, ((l+1)*M-1)*T , (l+1)*M*T, lambda x: 0, lambda x: ((l+1)*M*T-x))[0])
    
    def P15(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P15 = 0
            for l in range(0, K-1):
              resultado_P15 = resultado_P15 + P15l(l, M, T)
              #print("valor l",l,"resultado", resultado_P13)
            return resultado_P15
    
    def EC15(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC15 = 0
            def FC15(y, x):
                return (c*((l+1)*M*T-x-y)) *np.exp(-mi*x) * fx(x) * fy(y) * Rh((l+1)*M*T-(x+y))
            for l in range(0, K-1):
                resultado_EC15 = resultado_EC15 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P15l(l, M, T)) + (dblquad(  FC15, ((l+1)*M-1)*T , (l+1)*M*T, lambda x: 0, lambda x: ((l+1)*M*T-x))[0])
            return resultado_EC15
    
    def EL15(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL15 = 0
            for l in range(0, K-1):
                resultado_EL15 = resultado_EL15 + (((l+1)*M * T) * P15l(l, M, T))
            return resultado_EL15
    
    # CENÁRIO 16 - Defeito menor por choque, defeito maior e substituição em inspeção maior.
    
    def P16l(l, M, T):
        def FP16(y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l+1)*M*T-(z+y))
        return (dblquad(FP16, ((l+1)*M-1)*T , (l+1)*M*T, lambda z: 0, lambda z: ((l+1)*M*T-z))[0])
    
    def P16(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P16 = 0
            for l in range(0, K-1):
              resultado_P16 = resultado_P16 + P16l(l, M, T)
              #print("valor l",l,"resultado", resultado_P13)
            return resultado_P16
    
    def EC16(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC16 = 0
            def FC16(y, z):
                return (c*((l+1)*M*T-z-y)) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l+1)*M*T-(z+y))
            for l in range(0, K-1):
                resultado_EC16 = resultado_EC16 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P16l(l, M, T)) + (dblquad(FC16, ((l+1)*M-1)*T , (l+1)*M*T, lambda z: 0, lambda z: ((l+1)*M*T-z))[0])
            return resultado_EC16
    
    def EL16(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL16 = 0
            for l in range(0, K-1):
                resultado_EL16 = resultado_EL16 + (((l+1)*M * T) * P16l(l, M, T))
            return resultado_EL16
    
    
    # CENÁRIO 17 - Defeito menor por degradação, defeito maior e substituição em inspeção maior APÓS UM OU MAIS FALSO NEGATIVOS.
    
    def P17li(l, i, T):
        def FP17(y, x):
            return np.exp(-mi*x) * fx(x) * fy(y) * Rh((l+1)*M*T-(x+y))
        return (b**(M-i)) * (dblquad(FP17, (l*M+(i-1))*T , (l*M+i)*T, lambda x: (((l+1)*M*T)-T-x), lambda x: ((l+1)*M*T-x))[0])
    
    def P17(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P17 = 0
            for l in range(0, K-1):
                for i in range(1,M):
                  resultado_P17 = resultado_P17 + P17li(l, i, T)
            return resultado_P17
    
    def EC17(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC17 = 0
            def FC17(y, x):
                return (c*((l+1)*M*T-x-y)) * np.exp(-mi*x) * fx(x) * fy(y) * Rh((l+1)*M*T-(x+y))
            for l in range(0, K-1):
              for i in range(1,M):
                resultado_EC17 = resultado_EC17 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P17li(l, i, T)) + (b**(M-i)) * (dblquad(FC17, (l*M+(i-1))*T , (l*M+i)*T, lambda x: (((l+1)*M*T)-T-x), lambda x: ((l+1)*M*T-x))[0])
            return resultado_EC17
    
    def EL17(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL17 = 0
            for l in range(0, K-1):
              for i in range(1,M):
                resultado_EL17 = resultado_EL17 + (((l+1)*M * T) * P17li(l, i, T))
            return resultado_EL17
    
    
    # CENÁRIO 18 - Defeito menor por choque, defeito maior e substituição em inspeção maior.
    
    def P18li(l, i, T):
        def FP18(y, z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l+1)*M*T-(z+y))
        return (b**(M-i)) * (dblquad(FP18, (l*M+(i-1))*T , (l*M+i)*T, lambda z: (((l+1)*M*T)-T-z), lambda z: ((l+1)*M*T-z))[0])
    
    def P18(K, M, T):
        if K == 1:
          return 0
        else:
            resultado_P18 = 0
            for l in range(0, K-1):
                for i in range(1,M):
                  resultado_P18 = resultado_P18 + P18li(l, i, T)
            return resultado_P18
    
    def EC18(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EC18 = 0
            def FC18(y, z):
                return (c*((l+1)*M*T-z-y)) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh((l+1)*M*T-(z+y))
            for l in range(0, K-1):
              for i in range(1,M):
                resultado_EC18 = resultado_EC18 + ((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P18li(l, i, T)) + (b**(M-i)) * (dblquad(FC18, (l*M+(i-1))*T , (l*M+i)*T, lambda z: (((l+1)*M*T)-T-z), lambda z: ((l+1)*M*T-z))[0])
            return resultado_EC18
    
    def EL18(K, T, M):
        if K == 1:
          return 0
        else:
            resultado_EL18 = 0
            for l in range(0, K-1):
              for i in range(1,M):
                resultado_EL18 = resultado_EL18 + (((l+1)*M * T) * P18li(l, i, T))
            return resultado_EL18
    
    # CENÁRIO 19 -Defeito menor chega por degradação de e é substituído em inspeção maior em KMT
    
    def P19i(K, i, T):
        def FP19(x):
            return np.exp(-mi*x) * fx(x) * Ry(K*M*T-x)
        return (b**(M-i)) * (quad(FP19, (((K-1)*M+(i-1)))*T, (((K-1)*M+i))*T)[0])
    
    def P19(K, M, T):
        resultado_P19 = 0
        for i in range(1,M+1):
          resultado_P19 = resultado_P19 + P19i(K, i, T)
        return resultado_P19
    
    def EC19(K, T, M):
        resultado_EC19 = 0
        for i in range(1, M+1):
            resultado_EC19 = resultado_EC19 + ((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P19i(K, i, T))
        return resultado_EC19
    
    def EL19(K, T, M):
        resultado_EL19 = 0
        for i in range(1, M+1):
            resultado_EL19 = resultado_EL19 + ((K*M * T) * P19i(K, i, T))
        return resultado_EL19
    
    
    # CENÁRIO 20 - Defeito menor chega por choque de e é substituído em inspeção maior em KMT
    
    def P20i(K, i, T):
        def FP20(z):
            return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * Ry(K*M*T-z)
        return (b**(M-i)) * (quad(FP20, (((K-1)*M+(i-1)))*T, (((K-1)*M+i))*T)[0])
    
    def P20(K, M, T):
        resultado_P20 = 0
        for i in range(1,M+1):
          resultado_P20 = resultado_P20 + P20i(K, i, T)
        return resultado_P20
    
    def EC20(K, T, M):
        resultado_EC20 = 0
        for i in range(1, M+1):
            resultado_EC20 = resultado_EC20 + ((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P20i(K, i, T))
        return resultado_EC20
    
    def EL20(K, T, M):
        resultado_EL20 = 0
        for i in range(1, M+1):
            resultado_EL20 = resultado_EL20 + ((K*M * T) * P20i(K, i, T))
        return resultado_EL20
    
    # CENÁRIO 21 - Defeito menor por degradação na ith-1 inspeção, defeito maior e substituição em KMT
    
    def P21(K, M, T):
        def FP21 (y, x):
          return np.exp(-mi*x) * fx(x) * fy(y) * Rh(K*M*T-(x+y))
        return (dblquad(FP21, (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
    
    def EC21(K, T, M):
          resultado_EC21 = 0
          def FC21 (y, x):
            return (c*(K*M*T-x-y)) * np.exp(-mi*x) * fx(x) * fy(y) * Rh(K*M*T-(x+y))
          resultado_EC21 = ((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P21(K,M,T)) + (dblquad( FC21, (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
          return resultado_EC21
    
    def EL21(K, T, M):
          resultado_EL21 = 0
          resultado_EL21 = ((K*M*T) * P21(K,M,T))
          return resultado_EL21
    
    
    # CENÁRIO 22 - Defeito menor por choque na ith-1 inspeção, defeito maior e substituição em KMT
    
    def P22(K, M, T):
        def FP22 (y, z):
          return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh(K*M*T-(z+y))
        return (dblquad(FP22, (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
    
    def EC22(K, T, M):
        resultado_EC22 = 0
        def FC22 (y, z):
          return (c*(K*M*T-z-y)) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh(K*M*T-(z+y))
        resultado_EC22 = ((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P22(K, M, T)) + (dblquad(FC22, (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
        return resultado_EC22
    
    def EL22(K, T, M):
        resultado_EL22 = 0
        resultado_EL22 = ((K*M*T) * P22(K, M, T))
        return resultado_EL22
    
    # CENÁRIO 23 - Defeito menor por degradação, defeito maior após falso negativo e substituição em KMT após um ou mais falso negativo
    
    def P23i(K, i, T):
        def FP23 (y, x):
          return np.exp(-mi*x) * fx(x) * fy(y) * Rh(K*M*T-(x+y))
        return (b**(M-i)) * (dblquad(FP23, ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
    
    def P23(K,M,T):
        resultado_P23 = 0
        for i in range(1,M):
          resultado_P23 = resultado_P23 + P23i(K, i, T)
        return resultado_P23
    
    def EC23(K, T, M):
        resultado_EC23 = 0
        def FC23 (y, x):
          return (c*(K*M*T-x-y)) * np.exp(-mi*x) * fx(x) * fy(y) * Rh(K*M*T-(x+y))
        for i in range(1, M):
            resultado_EC23 = resultado_EC23 + ((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P23i(K, i, T)) + (b**(M-i)) * (dblquad(FC23, ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
        return resultado_EC23
    
    def EL23(K, T, M):
        resultado_EL23 = 0
        for i in range(1, M):
            resultado_EL23 = resultado_EL23 + ((K*M * T) * P23i(K, i, T))
        return resultado_EL23
    
    
    # CENÁRIO 24 - Defeito menor por choque, defeito maior após falso negativo e substituição em KMT após um ou mais falso negativo
    
    def P24i(K, i, T):
        def FP24 (y, z):
          return mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh(K*M*T-(z+y))
        #return (dblquad(FP12, (K*M-1)*T, K*M*T, lambda z: 0, lambda z: K*M*T-z)[0])
        return (b**(M-i)) * (dblquad(FP24, ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
    
    def P24(K, M, T):
        resultado_P24 = 0
        for i in range(1,M):
                resultado_P24 = resultado_P24 + P24i(K, i, T)
        return resultado_P24
    
    def EC24(K, T, M):
        resultado_EC24 = 0
        def FC24 (y, z):
          return (c*(K*M*T-z-y)) * mi*np.exp(-mi*z) * (np.exp(-(z/n1)**b1)) * fy(y) * Rh(K*M*T-(z+y))
        for i in range(1, M):
            resultado_EC24 = resultado_EC24 + ((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P24i(K, i, T)) + (b**(M-i)) * (dblquad(FC24, ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
        return resultado_EC24
    
    def EL24(K, T, M):
        resultado_EL24 = 0
        for i in range(1, M):
            resultado_EL24 = resultado_EL24 + ((K*M * T) * P24i(K, i, T))
        return resultado_EL24
    
    
    # CENÁRIO 25 - Componente sobrevive até KMT
    
    def P25(K,M,T):
        def FP25(x):
            return fx(x)
        return np.exp(-mi*K*M*T) * (quad(FP25, K*M*T, np.inf) [0])
    
    def EC25(K, T, M):
        resultado_EC25 = 0
        resultado_EC25 = ((K-1)*cb + (K*(M-1)*ci)+cr) * P25(K, M, T)
        return resultado_EC25
    
    def EL25(K, T, M):
        resultado_EL25 = 0
        resultado_EL25 = ((K*M*T) * P25(K, M, T))
        return resultado_EL25

  ####################### MEDIDAS DE INTERESSE #################################

    def EC(K, T, M,):
        return EC1(K, T, M,) + EC2(K, T, M) + EC3(K, T, M) + EC4(K, T, M) + EC5(K, T, M) + EC6(K, T, M) + EC7(K, T, M) + EC8(K, T, M) + EC9(K, T, M) + EC10(K, T, M) + EC11(K, T, M) + EC12(K, T, M) + EC13(K, T, M) + EC14(K, T, M) + EC15(K, T, M) + EC16(K, T, M) + EC17(K, T, M) + EC18(K, T, M) + EC19(K, T, M) + EC20(K, T, M) + EC21(K, T, M) + EC22(K, T, M) + EC23(K, T, M) + EC24(K, T, M) + EC25(K, T, M)
    
    def EL(K, T, M):
        return EL1(K, T, M) + EL2(K, T, M) + EL3(K, T, M) + EL4(K, T, M) + EL5(K, T, M) + EL6(K, T, M) + EL7(K, T, M) + EL8(K, T, M) + EL9(K, T, M) + EL10(K, T, M) + EL11(K, T, M) + EL12(K, T, M) + EL13(K, T, M) + EL14(K, T, M) + EL15(K, T, M) + EL16(K, T, M) + EL17(K, T, M) + EL18(K, T, M) + EL19(K, T, M) + EL20(K, T, M) + EL21(K, T, M) + EL22(K, T, M) + EL23(K, T, M) + EL24(K, T, M) + EL25(K, T, M)

    return EC(K,T,M)/EL(K,T,M)

def main():
    #criando 3 colunas
    col1, col2, col3= st.columns(3)
    foto = Image.open('randomen.png')
    #st.sidebar.image("randomen.png", use_column_width=True)
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    #O código abaixo centraliza e atribui cor
    st.markdown("<h2 style='text-align: center; color: #306754;'>A maintenance policy for major and minor inspections in a three-stage failure process subject to external shocks</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #F3F3F3; padding: 10px; text-align: center;">
          <p style="font-size: 20px; font-weight: bold;">Major and minor inspections in a two-phase delay-time process that considers external shocks</p>
          <p style="font-size: 15px;">By: Eugenio A. de S. Fischetti, Yan-Fu Li & Cristiano A. V. Cavalcante</p>
        </div>
        """, unsafe_allow_html=True)

    menu = ["Cost-rate", "Information", "Website"]
    
    choice = st.sidebar.selectbox("Select here", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insert the parameter values below:")
        
        global n1,b1,l_tx,b,ci,cb,cr,cf,c
        n1=st.number_input("Insert the scale parameter for the minor defect arrival by natural degradation (ηx\u2081)", min_value = 0.0, value = 3.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the minor defect arrival by degradation.")
        b1=st.number_input("Insert the shape parameter for the minor defect arrival by natural degradation (βx\u2081)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the minor defect arrival by degradation.")
        mi=st.number_input("Insert the shock arrival rate (μ)",min_value = 0.0, max_value=5.0, value = .5, help="This parameter indicates the shock arrival rate when the component is in good condition..")
        escala2=st.number_input("Insert the scale parameter for the major defect arrival by natural degradation (ηy\u2082)", min_value = 0.0, value = 3.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the major defect arrival by degradation.")
        forma2=st.number_input("Insert the shape parameter for the major defect arrival by natural degradation (βy\u2082)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the major defect arrival by degradation.")
        l_tx=st.number_input("Insert the rate of the exponential distribution for delay-time (λ)", min_value = 0.0, value = 2.0, help="This parameter defines the rate of the Exponential distribution, which governs the transition from the defective to the failed state of a component.")
        b=st.number_input("Insert the false-negative probability (\u03B5)", min_value = 0.0, max_value=1.0, value = 0.15, help="This parameter represents the probability of not indicating a defect during inspection when, in fact, it does exist.")
        ci=st.number_input("Insert cost of inspection (Ci)", min_value = 0.0, value = 0.1, help="This parameter represents the cost of conducting a minor inspection.")
        cb=st.number_input("Insert cost of inspection (Cb)", min_value = 0.0, value = 0.3, help="This parameter represents the cost of conducting a major inspection.")
        cr=st.number_input("Insert cost of replacement (inspections and age-based) (Cr)", min_value = 0.0, value = 1.0, help="This parameter represents the cost associated with preventive replacements, whether performed during inspections or when the age-based threshold is reached.")
        cf=st.number_input("Insert cost of failure (Cf)", min_value = 0.0, value = 10.0, help="This parameter represents the replacement cost incurred when a component fails.")
        c = st.number_input("Insert cost per major defective time unit (c)", min_value=0.0, value=10.0)

        col1, col2 = st.columns(2)
        

        st.subheader("Policy Parameters (K-M-T Model)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            K = st.number_input(
                "Number of major blocks (K)", 
                min_value=1, value=3, step=1,
                help="Total number of major inspection blocks before replacement"
            )
        
        with col2:
            M = st.number_input(
                "Minor inspections per block (M)", 
                min_value=1, value=4, step=1,
                help="Number of minor inspections within each major block"
            )
        
        with col3:
            T = st.number_input(
                "Interval between inspections (T)", 
                min_value=0.1, value=.5, step=0.5,
                format="%.1f",
                help="Fixed time interval between consecutive minor inspections"
            )
        
        # Visualização da estrutura da política
        st.markdown(f"""
        **Policy Structure:**  
        `Preventive replacement at: {K*M*T:.1f} time units`  
        (K = {K} blocks × M = {M} inspections × T = {T} interval)
        """)

        # Botão de cálculo
        if st.button("Calculate Cost-Rate", key="kmt_calc"):
            try:
                with st.spinner("Analyzing..."):
                    cost_rate = KMT(K, M, T, b1, n1, forma2, escala2, l_tx, b, ci, cb, cr, cf, c, mi)
                    # Mostrando o resultado (AGORA USANDO cost_rate)
                    st.metric(
                        label="Cost-Rate", 
                        value=f"{cost_rate:.4f}", 
                        help="Average cost per unit time"
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("""
                    Troubleshooting Tips:
                    1. Reduce K/M values
                    2. Increase inspection interval T
                    3. Verify parameter limits
                    """)
         
    if choice == menu[1]:
        st.header(menu[1])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>This application calculates the cost ratio for a maintenance model based on a two-phase time-delay approach for single-component systems subject to natural degradation and external shocks. The proposed inspection and replacement (I/R) policy aims to optimize inspection intervals and maintenance actions, considering the possibility of imperfect inspections..</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>The app computes the cost-rate for a specific solution—defined by the number of major inspections until preventive replacement (K), number of minor inspections up to the major inspection (included) (M);  and the time interval between two consecutive inspections; (T).</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>For further questions or information on finding the optimal solution, please contact one of the email addresses below.</h6>", unsafe_allow_html=True)
        
        st.write('''

e.a.s.fischetti@random.org.br

liyanfu@tsinghua.edu.cn

c.a.v.cavalcante@random.org.br

''')    
    if choice==menu[2]:
        st.header(menu[2])
        
        st.write('''The Research Group on Risk and Decision Analysis in Operations and Maintenance was created in 2012 
                 in order to bring together different researchers who work in the following areas: risk, maintenance a
                 nd operation modelling. Learn more about it through our website.''')
        st.markdown('[Click here to be redirected to our website](https://sites.ufpe.br/random/#page-top)',False)        
if st._is_running_with_streamlit:
    main()
else:
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
