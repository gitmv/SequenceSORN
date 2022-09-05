from PymoNNto import *
import matplotlib.pyplot as plt


#plt.scatter([25,30,35,40, 45,50, 55,50,20,15,10,5],[4.937856371015798,4.938395473223446,4.941162023024411,4.933519252712802,4.939664728053855,4.941525735888286,4.938454326902593,4.93847460324344, 4.939998524761805, 4.940388482592516, 4.939559388011622, 7.231853232628977], label='4s')

#plt.scatter([3,4,5,6,7,8,9,10,15,20,25,30,35,40],[6.523999427157182,5.8413627438135425,6.63827173369897,7.881901943053844,7.9645617766136105,7.172966344891123,7.774978510158176,7.672365480553815,3.800265606495106,3.79680646933705,3.802304824481012,3.798419483238679,3.80293814666511,3.8012605515653783], label='3s')

#plt.scatter([3], [6.523999427157182], label='3s')
#plt.scatter([4], [5.8413627438135425], label='3s')
#plt.scatter([5], [6.63827173369897], label='3s')
#plt.scatter([6], [7.881901943053844], label='3s')
#plt.scatter([7], [7.9645617766136105], label='3s')
#plt.scatter([8], [7.172966344891123], label='3s')
#plt.scatter([9], [7.774978510158176], label='3s')
#plt.scatter([10], [7.672365480553815], label='3s')
#plt.scatter([15], [3.800265606495106], label='3s')
#plt.scatter([20], [3.79680646933705], label='3s')
#plt.scatter([25], [3.802304824481012], label='3s')
#plt.scatter([30], [3.798419483238679], label='3s')
#plt.scatter([35], [3.80293814666511], label='3s')
#plt.scatter([40], [3.8012605515653783], label='3s')


plt.scatter([70,80,90,100,110,120,130,140,150,160,170,180,190,200], [6.640798331001812,7.03259135276443,6.80826101930977,6.501052825111525,6.244692517487972,7.055451833203204,6.398650357125684,6.855678635421277,6.715132251321035,6.946591042859746,6.332428057678019,6.968184134392905,6.095721478457292,6.6531626330161115], label='stdp-inh')

#plt.scatter([70], [6.640798331001812], label='stdp-inh')
#plt.scatter([80], [7.03259135276443], label='stdp-inh')
#plt.scatter([90], [6.80826101930977], label='stdp-inh')
#plt.scatter([100], [6.501052825111525], label='stdp-inh')
#plt.scatter([110], [6.244692517487972], label='stdp-inh')
#plt.scatter([120], [7.055451833203204], label='stdp-inh')
#plt.scatter([130], [6.398650357125684], label='stdp-inh')
#plt.scatter([140], [6.855678635421277], label='stdp-inh')
#plt.scatter([150], [6.715132251321035], label='stdp-inh')
#plt.scatter([160], [6.946591042859746], label='stdp-inh')
#plt.scatter([170], [6.332428057678019], label='stdp-inh')
#plt.scatter([180], [6.968184134392905], label='stdp-inh')
#plt.scatter([190], [6.095721478457292], label='stdp-inh')
#plt.scatter([200], [6.6531626330161115], label='stdp-inh')


plt.scatter([80,100,120,140,160,180,200], [6.782841718633819,6.322812524288367,5.22646737158857,3.7989632775254787,4.106028414419509,4.950225248559938,6.630217120393806], label='stdp-inh_-')

#plt.scatter([80], [6.782841718633819], label='stdp-inh_-')
#plt.scatter([100], [6.322812524288367], label='stdp-inh_-')
#plt.scatter([120], [5.22646737158857], label='stdp-inh_-')
#plt.scatter([140], [3.7989632775254787], label='stdp-inh_-')
#plt.scatter([160], [4.106028414419509], label='stdp-inh_-')
#plt.scatter([180], [4.950225248559938], label='stdp-inh_-')
#plt.scatter([200], [6.630217120393806], label='stdp-inh_-')

plt.legend()

plt.show()