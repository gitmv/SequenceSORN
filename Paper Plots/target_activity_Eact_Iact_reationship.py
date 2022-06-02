from PymoNNto import *
import matplotlib.pyplot as plt

xparam = 'ta'
yparam = 'exp'
zparam = 's'


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def drawGroup(name, b=0.0):
    folder = get_data_folder() + '/Evolution_Project_Clones/' + name + '/Data'
    smg = StorageManagerGroup(name, data_folder=folder)
    data = smg.get_multi_param_dict(params=[xparam, yparam, zparam, 'score'] ,remove_None=True)

    mi=np.min(data['score'])
    ma=np.max(data['score'])
    c=(data['score']-mi)/(ma-mi)
    c=[(1.0-d,d,b,0.3) for d in c]
    ax.scatter(data[xparam]/100.0, data[yparam], data[zparam], c=c)


drawGroup('abc72_1500_evo', 1.0)

drawGroup('abc_1500_evo', 0.0)
drawGroup('abci_1500_evo', 1.0)
drawGroup('abc5_1500_evo', 0.0)
drawGroup('abc2_1500_evo', 1.0)

ax.set_xlabel('h')#xparam
ax.set_ylabel('b')#yparam
ax.set_zlabel('c')#zparam

additional_points = np.array([
    [1.2773029870935577, 0.7133902130957764, 21.09512032222327],

    #[1.316,0.6934,19.2557], #abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[](){}<>.
    [3.0,0.614,20.0],       #F|B|P
    [3.988,0.5,15.27],      #abcdefghijklmnopqrstu.
    #[5.419,0.437,9.87],     #abcdefghi. ###
    [8.76, 0.347, 7.43],    #abcdefghi.
    #[12.9, 0.31, 5.846],    #???
    [17.02, 0.28, 5.45],    #abc.
    #[19.128, 0.267, 5.7558],    #abc. ###

    #[27.547498574447054, 0.2809896419670679, 5.527288999306307], #.
    #[40, 0.2809896419670679, 5.527288999306307],  # . ###
    #[50, 0.2809896419670679, 5.527288999306307],  # . ###

    [30.341837621320494, 0.22683853231902354, 5.5676308850895], #.
    #[40, 0.22683853231902354, 5.5676308850895],  # . ###
    #[50, 0.22683853231902354, 5.5676308850895],  # . ###



    #[20, 0.28, 5.5],    #???
    #[24, 0.28, 5.5],    #???
    #[27, 0.28, 5.5],    #???
    #[30, 0.28, 5.5],    #???
])

x = additional_points[:, 0]/100.0
y = additional_points[:, 1]
z = additional_points[:, 2]

ax.scatter(x,y,z, c=[(0.0, 0.0, 1.0, 1.0)], s=100)

pxy = np.poly1d(np.polyfit(x, y, 2))
pxz = np.poly1d(np.polyfit(x, z, 2))

def y_fit(d):
    return 0.01/d+0.22

def z_fit(d):
    return 0.4/d+3.6
#return 28/d+4

x=np.array(np.arange(0.02, 0.3, 0.001))
plt.plot(x, y_fit(x), z_fit(x))

plt.show()






#for p in additional_points:
#    ax.scatter(p[0], p[1], p[2], c=[(0.0, 0.0, 1.0, 1.0)], s=100)

#ax.scatter([3.0], [0.614], [20.0], c=[(0.0,0.0,1.0,1.0)], s=100)#F|B|M
#ax.scatter([3.988351708556492], [0.5004156607096197], [15.274553117050054], c=[(0.0,0.0,1.0,1.0)], s=100)#abc26
#ax.scatter([5.419513086214809], [0.4375721000971742], [9.873254753125517], c=[(0.0,0.0,1.0,1.0)], s=100)#abci

#additional_points = np.array([[3.0,0.614,20.0], [3.988,0.5,15.27], [5.419,0.437,9.87]])
#additional_points_mean = additional_points.mean(axis=0)
#uu, dd, vv = np.linalg.svd(additional_points - additional_points_mean)
#print(uu,dd,vv)
#linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
#linepts += additional_points_mean
#ax.plot(linepts[:,0], linepts[:,1], linepts[:,2])


#v1 = linepts[0]
#v2 = linepts[1]-linepts[0]
#v2 = v2/v2[0]
#v1 = v1-v2*v1[0]
#p=np.array([v1,v1+v2])
#ax.plot(p[:,0],p[:,1],p[:,2])

#print('b =', v1[1],'+ h * ',v2[1])
#print('c =', v1[2],'+ h * ',v2[2])

#ax.plot([v1[0],v1[0]+10*v2[0]], [v1[1], v1[1]+10*v2[1]], [v1[2], v1[2]+10*v2[2]])

#ax.scatter([6.025], [0.348], [7.93], c=[(0.0, 0.0, 0.0, 1.0)], s=100)  # F|B|M

