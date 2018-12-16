import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import CF_2D_3D_views_newdata
from CF_2D_3D_views_newdata import min_minID_segment, min_minID
import mpl_toolkits.mplot3d as m3d   
from numpy.linalg import lstsq
import math
from math import acos
from math import sqrt
from math import pi

def read_3d_data():
    f = open("bp0-1.txt", "r")
    lines = f.readlines()[1:]
    f.close()
    x = []
    y = []
    z = []
    for line in lines:
        parts = line.split()
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        z.append(float(parts[2]))
    return x,y,z
    
def get_boundary_graph():
    seq_peak,seq_peak_id = min_minID(x)
    fig = plt.figure()
    base = np.arange(1,len(x)+1,1)
    plt.plot(base,x,'ro-')
    plt.plot((1,len(x)),(np.mean(x),np.mean(x)), 'b--')
    for i in range(0, len(seq_peak_id)):
        plt.plot((seq_peak_id[i],seq_peak_id[i]),(min(x)-2,max(x)+2), 'b-')
    plt.show()

def plane_fitting_basic(x,y,z):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])

    X,Y = np.meshgrid(np.arange(int(round(min(x))-1), int(round(max(x))+1), 1), np.arange(int(round(min(y))-1), int(round(max(y))+1), 1))
    XX = X.flatten()
    YY = Y.flatten()

    data = np.stack((x, y, z), axis=-1)
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,res,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    Z = C[0]*X + C[1]*Y + C[2]    
    ax.scatter(x, y, z, marker='o',c='r') 
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    plt.show()
    return C,res,Z
    
def plane_fitting_cycles(x,y,z):
    x_buff = []
    y_buff = []
    z_buff = []
    count = 0
    seq_peak,seq_peak_id = min_minID(x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    sct_points = None
    sct_plate= None

    for i in range(0, len(x)):
        if(count < len(seq_peak_id)-1):
            if(i <= seq_peak_id[count]):
                x_buff.append(x[i])  
                y_buff.append(y[i]) 
                z_buff.append(z[i]) 
                sct_points = ax.scatter(x[i], y[i], z[i], marker='o',c='r')  
                plt.pause(0.01)                  
                plt.show()
            else:
                X,Y = np.meshgrid(np.arange(int(round(min(x_buff))-1), int(round(max(x_buff))+1), 1), np.arange(int(round(min(y_buff))-1), int(round(max(y_buff))+1), 1))
                XX = X.flatten()
                YY = Y.flatten()
           
                data_new = np.stack((x_buff, y_buff, z_buff), axis=-1)
                A = np.c_[data_new[:,0], data_new[:,1], np.ones(data_new.shape[0])]
                C,res,_,_ = scipy.linalg.lstsq(A, data_new[:,2])    # coefficients
    
                Z = C[0]*X + C[1]*Y + C[2]
                res_ave = res/len(x_buff)
                print(res_ave)    
                
                if sct_plate is not None:
                    sct_plate.remove()
                    sct_points.remove()
            
                sct_plate = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
                
                '''
                if sct_points is not None:
                    sct_points.remove()
                '''
                sct_points = ax.scatter(data_new[:,0], data_new[:,1], data_new[:,2], c='r', s=50)      
                plt.show()
                plt.pause(0.1)    
               
                x_buff = []
                y_buff = []
                z_buff = []
                x_buff.append(x[i])
                y_buff.append(y[i])
                z_buff.append(z[i])
                count = count+1
                                  
def plane_fitting_dynamic(x,y,z):  
    x_buff_temp = []
    y_buff_temp = []
    z_buff_temp = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])

    sct_points = None
    sct_plate= None
    res_ave_arr = []
    angle_arr = []
    for i in range(0, len(x)):
        x_buff_temp.append(x[i])  
        y_buff_temp.append(y[i]) 
        z_buff_temp.append(z[i]) 
        sct_points = ax.scatter(x[i], y[i], z[i], marker='o',c='r')  
        plt.pause(0.001)                  
        plt.show()
        
        if(i>14):
            X,Y = np.meshgrid(np.arange(int(round(min(x))-1), int(round(max(x))+1), 1), np.arange(int(round(min(y))-1), int(round(max(y))+1), 1))
            XX = X.flatten()
            YY = Y.flatten()

            data_new = np.stack((x_buff_temp, y_buff_temp, z_buff_temp), axis=-1)
            A = np.c_[data_new[:,0], data_new[:,1], np.ones(data_new.shape[0])]
            C,res,_,_ = scipy.linalg.lstsq(A, data_new[:,2])    # coefficients
    
            Z = C[0]*X + C[1]*Y + C[2]
            if(i == 15):
                v_temp = [C[0],C[1],-1]
            else:
                v_cur = [C[0],C[1],-1]
                angle=np.arccos(np.dot(v_temp,v_cur)/(np.linalg.norm(v_temp)*np.linalg.norm(v_cur)))
                angle_arr.append(angle)
                v_temp = [C[0],C[1],-1]
                print('angle is %f', angle)
                
            res_ave = res/len(x_buff_temp)
            res_ave_arr.append(res_ave)
            print(res_ave)    
            
            if sct_plate is not None:
                sct_plate.remove()
          
            sct_plate = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        return res_ave_arr, angle_arr

def line_fitting(x,y,z):
    data1 = np.stack((x, y, z), axis=-1) 
    datamean = data1.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data1 - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # Now generate some points along this best fit line, for plotting.

    # I use -7, 7 since the spread of the data is roughly 14
    # and we want it to have mean 0 (like the points we did
    # the svd on). Also, it's a straight line, so we only need 2 points.
    linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean

    # Verify that everything looks right.
    ax = m3d.Axes3D(plt.figure())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter3D(*data1.T)
    ax.plot3D(*linepts.T)
    plt.show() 
    
def residual_average_trend(res_ave_arr):
    plt.figure()
    plt.grid(True)
    plt.title('Average Residual Trend')
    plt.xlabel('Points')
    plt.ylabel('Average Residual')
    res_ave_base = np.arange(1,len(res_ave_arr)+1,1) 
    plt.plot(res_ave_base,res_ave_arr,'bo-')
    plt.show()

def angle_of_plane_trend(angle_arr):              
    plt.figure()
    plt.grid(True)
    plt.title('Angle of normal vectors')
    plt.xlabel('Points')
    plt.ylabel('arccos')
    angle_base = np.arange(1,len(angle_arr)+1,1) 
    plt.plot(angle_base,angle_arr,'bo-')
    plt.show()

def plane_formal_exp(C):  
    '''
    build equation set
    Z = 0*X + 0*Y + 1
    C0*X + C1*Y + C2*Z = 1
    (xp-x)/C0=t
    (yp-y)/C1 = t
    (zp-z)/C2=t
    '''
    C2= 1.0/C[2]
    C0= -C[0]*C2
    C1 = -C[1]*C2
    return C0,C1,C2
    
def points_projection(x,y,z,C):
    proj_x = []
    proj_y = []
    proj_z = []
    C0,C1,C2 = plane_formal_exp(C)
    A=np.array([[1.0,0,0,-C0],[ 0,1.0,0,-C1],[0,0,1.0,-C2],[C0,C1,C2,0]])
    for i in range(1,len(x)):
        xx = x[i]
        yy = y[i]
        zz = z[i]
        b = np.array([xx,yy,zz,1])
        r = np.linalg.solve(A,b)
        proj_x.append(r[0])
        proj_y.append(r[1])
        proj_z.append(r[2])
    return proj_x, proj_y, proj_z
          

def points_proj_graph(proj_x, proj_y, proj_z, Z):
   
    x,y,z = read_3d_data()
    
    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    X,Y = np.meshgrid(np.arange(int(round(min(x))-1), int(round(max(x))+1), 1), np.arange(int(round(min(y))-1), int(round(max(y))+1), 1))
    XX = X.flatten()
    YY = Y.flatten()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
    ax.scatter(data[:,0], data[:,1], 0, c='r', s=50)
    ax.scatter(proj_x, proj_y, proj_z, c='b', s=50)
    plt.show()

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  val = math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
  angle = val*180/pi
  if(angle>90):
      return angle - 90
  else:
      return angle
      
x,y,z = read_3d_data()
#plane_fitting_cycles(x,y,z)
#plane_fitting_dynamic(x,y,z)


data = np.stack((x, y, z), axis=-1)
C,res,Z = plane_fitting_basic(x,y,z)
C0,C1,C2 = plane_formal_exp(C)
print(C0,C1,C2)
print(res,res/len(x))
c_new = [C0,C1,C2]
c_z = [0,0,1]
angle = angle(c_new,c_z)
print(angle)


#proj_x, proj_y, proj_z = points_projection(x,y,z,C)
#points_proj_graph(proj_x, proj_y, proj_z,Z)
plane_fitting_basic(x,y,z)
  


