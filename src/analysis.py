import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import colorConverter
from scipy.signal import butter, sosfiltfilt, ricker, cwt, correlate
from scipy import stats
import networkx as nx

# Filtration images
Tzac1=500  # zacetek vizualizacije v izhodnih slikah, v sekundah
Tkon1=1000  # konec vizualizacije v izhodnih slikah, v sekundah

# Binarization images
Tzac2=500
Tkon2=1000

# Filtration parameters
FAST_lowBF=0.04  # spodnja frekvenca pri hitri komponenti (tipično 0.01-0.1)
FAST_highBF=0.4  # zgornja frekvenca pri hitri komponenti (tipično 0.2-2.0), zgornja meja naj bo manjsa od polovice samplinga (npr. sampling=1->0.49)

SLOW_lowBF=0.001  # spodnja frekvenca pri počasni komponenti (tipično 0.001-0.005)
SLOW_highBF=0.005  # zgornja frekvenca pri počasni komponenti (tipično 0.02-0.1)



# Binarization parameters
PB=4  # pricakovana dolzina oscilacij
act_slope=1.4   # mejni odvod  (med 1 in 2)
amp_faktor=1.4 # amplitudni faktor (med 1 in 2)
step=1   #int(round(sampling/2.0))

CORR_SLOW = 0.9
CORR_FAST = 0.75

class Analysis(object):
    """docstring for Analysis."""

    def __init__(self, data_name, sampling_name, positions_name):
        data = np.loadtxt(data_name)
        self.__time = data[:-1,0]
        self.__signal = data[:-1,1:]
        self.__sampling=int(np.loadtxt(sampling_name))
        self.__positions = np.loadtxt(positions_name)

        self.__data_points = len(self.__time)
        self.__number_of_cells = len(self.__signal[0])

        self.__filtered_slow = False
        self.__filtered_fast = False

        self.__binarized_slow = False
        self.__binarized_fast = False

        self.__excluded = False

        self.__G_slow = False
        self.__G_fast = False

    def import_slow(self, path):
        self.__filtered_slow = np.loadtxt(path, delimiter=",")

    def import_fast(self, path):
        self.__filtered_fast = np.loadtxt(path, delimiter=",")

    def import_binarized_slow(self, path):
        self.__binarized_slow = np.loadtxt(path, delimiter=",").astype(int)

    def import_binarized_fast(self, path):
        self.__binarized_fast = np.loadtxt(path, delimiter=",").astype(int)

    def import_excluded(self, path):
        self.__excluded = np.loadtxt(path, dtype=int)

    def save_slow(self, path):
        if self.__filtered_slow is False:
            raise ValueError("No filtered data!")
        np.savetxt(path, self.__filtered_slow, delimiter=",")

    def save_fast(self, path):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        np.savetxt(path, self.__filtered_fast, delimiter=",")

    def save_binarized_slow(self, path):
        np.savetxt(path, self.__binarized_slow, delimiter=",", fmt="%d")

    def save_binarized_fast(self, path):
        np.savetxt(path, self.__binarized_fast, delimiter=",", fmt="%d")

    def filter(self):
        self.__filtered_slow = np.zeros((self.__data_points, self.__number_of_cells))
        self.__filtered_fast = np.zeros((self.__data_points, self.__number_of_cells))

        for i in range(self.__number_of_cells):
            self.__filtered_slow[:,i] = self.__bandpass(self.__signal[:,i], SLOW_lowBF, SLOW_highBF)
            self.__filtered_fast[:,i] = self.__bandpass(self.__signal[:,i], FAST_lowBF, FAST_highBF)

    def __bandpass(self, data, lowcut, highcut, order=5):
        nyq = 0.5*self.__sampling
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        y = sosfiltfilt(sos, data)
        return y

    def plot_filtered(self, directory):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        Tzac=int(Tzac1*self.__sampling)
        Tkon=int(Tkon1*self.__sampling)
        for i in range(self.__number_of_cells):
            print(i)
            plt.figure(figsize=(12.5,5.0))
            plt.subplot(221)
            plt.plot(self.__time,self.__signal[:,i],linewidth=0.5,color='dimgrey')
            plt.title(r'trace $ %ld $: FAST: %.3f --> %.3f  SLOW: %.4f --> %.4f'%(float(i)/1.0,(FAST_lowBF),(FAST_highBF),(SLOW_lowBF),(SLOW_highBF)),fontsize=8)
            plt.rc('font', size=6)
            plt.subplot(222)
            miny=np.min(self.__signal[(Tzac):(Tkon),i])
            maxy=np.max(self.__signal[(Tzac):(Tkon),i])
            plt.plot(self.__time,self.__signal[:,i],linewidth=0.67,color='dimgrey')
            plt.xlim([Tzac1,Tkon1])
            plt.ylim([miny-0.05,maxy+0.05])
            plt.subplot(223)
            miny=np.min(self.__filtered_fast[Tzac:Tkon,i])
            maxy=np.max(self.__filtered_fast[Tzac:Tkon,i])
            plt.plot(self.__time,self.__filtered_fast[:,i],linewidth=0.67,color='dimgrey')
            plt.xlim([Tzac1,Tkon1])
            plt.ylim([miny*1.05-0.05,maxy*1.05+0.05])
            plt.xlabel('time (s)', fontsize=6)
            plt.subplot(224)
            miny=np.min(self.__filtered_slow[Tzac:Tkon,i])
            maxy=np.max(self.__filtered_slow[Tzac:Tkon,i])
            plt.plot(self.__time,self.__filtered_slow[:,i],linewidth=0.67,color='dimgrey')
            plt.xlim([Tzac1,Tkon1])
            plt.ylim([miny*1.05-0.05,maxy*1.05+0.05])
            plt.xlabel('time (s)', fontsize=6)

            plt.savefig("{0}/{1}.pdf".format(directory, i), dpi=200, bbox_inches='tight')
            plt.close()

    def __smooth(self, data, N):
        result=np.zeros((self.__data_points,self.__number_of_cells))

        for rep in range(self.__number_of_cells):
            zamik=int(np.round(N/2))
            for i in range(zamik-1,self.__data_points-zamik,1):
                avg=0.0
                for j in range(i-zamik+1,i+zamik,1):
                    avg=avg+data[j][rep]
                result[i][rep]=avg/(float)(N)

            for i in range(0,zamik,1):
                result[i][rep]=np.mean(data[i:(i+zamik),rep])
            for i in range(self.__data_points-1*zamik,self.__data_points,1):
                result[i][rep]=np.mean(data[(i-zamik):i,rep])

        return result

    def smooth_fast(self, repeats, N):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        for i in range(repeats):
            self.__filtered_fast = self.__smooth(self.__filtered_fast, N)

    def binarize_slow(self):
        self.__binarized_slow = np.zeros((self.__data_points, self.__number_of_cells))
        for cell in range(self.__number_of_cells):
            signal = self.__filtered_slow[:,cell]
            self.__binarized_slow[:,cell] = np.heaviside(np.gradient(signal), 0)
            extremes = []
            for i in range(1, self.__data_points):
                if self.__binarized_slow[i,cell]!=self.__binarized_slow[i-1,cell]:
                    extremes.append(i)

            up = self.__binarized_slow[0,cell]
            counter = 0
            for e in extremes:
                interval = e-counter
                for phi in range(6):
                    lower = counter+int(np.floor(interval/6*phi))
                    higher = counter+int(np.floor(interval/6*(phi+1)))
                    add = 1 if up else 7
                    self.__binarized_slow[lower:higher,cell] = phi+add
                up = (up+1)%2
                counter = e

            # Erase values (from 0 to first extreme) and (last extreme to end)
            self.__binarized_slow[0:extremes[0],cell] = 0
            self.__binarized_slow[extremes[-1]:,cell] = 0
            self.__binarized_slow.astype(int)


    def __binarize(self, data):
        signal=data[:,:]
        PB_i = [int(PB) for i in range(self.__number_of_cells)]
        odrez=int(round(0.05*self.__data_points))
        varsig=np.zeros(self.__number_of_cells)
        for rep in range(self.__number_of_cells):
            varsig[rep]=np.std(signal[odrez:self.__data_points-odrez,rep])

        derser1=[]
        derser2=[]
        for rep in range(self.__number_of_cells):
            derser1.append([])
            derser2.append([])
            for i in range(self.__data_points-PB_i[rep]-2):
                derser1[rep].append((((signal[i+1][rep]+signal[i+2][rep])/2.0)-signal[i][rep])/step)
                derser2[rep].append((((signal[i+2][rep]+signal[i+3][rep]+signal[i+4][rep])/3.0)-signal[i][rep])/(2*step))

        varderser1=np.zeros(self.__number_of_cells)
        varderser2=np.zeros(self.__number_of_cells)
        for rep in range(len(derser1)):
            varderser1[rep]=np.std(derser1[rep][:])
            varderser2[rep]=np.std(derser2[rep][:])

        binsig=np.zeros((self.__data_points, self.__number_of_cells))

        nnact=np.zeros(self.__number_of_cells, int)
        tact=[]
        for rep in range(self.__number_of_cells):
            tact.append([])
            for i in range(len(data)-PB_i[rep]-2):
                if (PB_i[rep]<=5):
                    slp1=(((signal[i+1][rep]+signal[i+2][rep])/2.0)-signal[i][rep])/step
                    slp2=(((signal[i+2][rep]+signal[i+3][rep]+signal[i+4][rep])/3.0)-signal[i][rep])/(2*step)
                else:
                    slp1=(((signal[i+2][rep]+signal[i+4][rep])/2.0)-signal[i][rep])/(2*step)
                    slp2=(((signal[i+4][rep]+signal[i+6][rep]+signal[i+8][rep])/3.0)-signal[i][rep])/(4*step)
                ok0 = 0 # preverja odvode
                if ((slp1>act_slope*varderser1[rep]) or (slp2>act_slope*varderser2[rep])): ok0=1
                ok1 = 0 # preverja, ce so v nadaljevanju dovolj visoki
                for p in range(PB_i[rep]):
                    if (signal[i+p][rep]>amp_faktor*varsig[rep]): ok1+=1
                if ( (ok0>0) and (ok1>1) ): #and (signal[i][rep]>(-0.5*varsig[rep])) ):
                    ok2 = 1
                    if (nnact[rep]<1):
                        ok2=1
                    else:
                        for iii in range(int(1.25*step*PB_i[rep])):
                            if ( np.abs(tact[rep][nnact[rep]-1]-(i+iii))<step*PB_i[rep]):
                                ok2=0
                                break
                    if (ok2==1):
                        nnact[rep]+=1
                        tact[rep].append(i)

        maxser=[]
        tmax=[]
        minser=[]
        tmin=[]
        for rep in range(self.__number_of_cells):
            maxser.append([])
            tmax.append([])
            maxser[rep]=np.zeros(len(tact[rep]))-10000
            for kk in range(len(tact[rep])):
                tmax[rep].append(tact[rep][kk])
            for ii in range(nnact[rep]):
                for i in range((tact[rep][ii]+1),(tact[rep][ii]+2*step*PB_i[rep]),1):
                    if (i<len(signal)-3*step*PB_i[rep]):
                        if (signal[i][rep]>maxser[rep][ii]):
                            maxser[rep][ii]=signal[i][rep]
                            tmax[rep][ii]=i
            minser.append([])
            tmin.append([])
            minser[rep]=np.zeros(len(tmax[rep]))+10000
            for kk in range(len(tmax[rep])):
                tmin[rep].append(tmax[rep][kk])
            for ii in range(nnact[rep]):
                for i in range((tmax[rep][ii]+1),(tmax[rep][ii]+3*step*PB_i[rep]),1):
                    if (i<len(signal)-3*step*PB_i[rep]):
                        if (signal[i][rep]<minser[rep][ii]):
                            minser[rep][ii]=signal[i][rep]
                            tmin[rep][ii]=i

        tfin=[]
        for rep in range(self.__number_of_cells):
            tfin.append([])
            for kk in range(len(tmax[rep])):
                tfin[rep].append(tmax[rep][kk])
            for ii in range(nnact[rep]):
                for i in range((tmax[rep][ii]+1),(tmin[rep][ii]),1):
                    if (signal[i][rep]<((0.5*maxser[rep][ii]+0.5*minser[rep][ii]))):
                        tfin[rep][ii]=i
                        break

        for rep in range(self.__number_of_cells):
            ii=0
            nobin=0
            for i in range(self.__data_points-PB_i[rep]-2):
                if ( (i>=tact[rep][ii]) and (i<=tfin[rep][ii]) and (nobin==0) ):
                    binsig[i][rep]=1
                if ((i>tmin[rep][ii]) and (ii<len(tact[rep])-1) ):
                    ii+=1
                if ( (i>tact[rep][ii]) and (ii==(len(tact[rep]))) ): nobin=1

        return binsig

    def binarize_fast(self):
        if self.__filtered_fast is False:
            raise ValueError("No filtered data!")
        self.__binarized_fast = self.__binarize(self.__filtered_fast).astype(int)

    def plot_binarized(self, directory):
        if self.__binarized_slow is False or self.__binarized_fast is False:
            raise ValueError("No filtered data!")
        # normiranje signala med 0 in 1
        norm_slow = self.__filtered_slow[:,:]
        norm_fast = self.__filtered_fast[:,:]
        for rep in range(self.__number_of_cells):
            norm_slow[:,rep]=((norm_slow[:,rep]-min(norm_slow[:,rep]))/(max(norm_slow[:,rep])-min(norm_slow[:,rep])))*11
            norm_fast[:,rep]=(norm_fast[:,rep]-min(norm_fast[:,rep]))/(max(norm_fast[:,rep])-min(norm_fast[:,rep]))

        for rep in range(self.__number_of_cells):
            print(rep)
            plt.subplot(221)
            plt.plot(self.__time,norm_slow[:,rep],linewidth=0.4,color='dimgrey')
            plt.plot(self.__time,self.__binarized_slow[:,rep],linewidth=0.5,color='red')
            plt.ylim([-0.05,13.0])
            plt.title(r'trace $ %ld $'%(float(rep)/1.0),fontsize=8)
            plt.rc('font', size=6)
            plt.subplot(223)
            plt.plot(self.__time,norm_slow[:,rep],linewidth=0.4,color='lightgrey')
            plt.plot(self.__time,norm_slow[:,rep],linewidth=0.6,color='black')
            plt.plot(self.__time,self.__binarized_slow[:,rep],linewidth=0.5,color='red')
            plt.xlim([Tzac2,Tkon2])
            plt.ylim([-0.05,13.0])
            plt.rc('font', size=6)
            plt.xlabel('time (s)', fontsize=6)

            plt.subplot(222)
            plt.plot(self.__time,norm_fast[:,rep],linewidth=0.4,color='dimgrey')
            plt.plot(self.__time,self.__binarized_fast[:,rep],linewidth=0.5,color='red')
            plt.ylim([-0.05,1.05])
            plt.title(r'trace $ %ld $'%(float(rep)/1.0),fontsize=8)
            plt.rc('font', size=6)
            plt.subplot(224)
            plt.plot(self.__time,self.__binarized_fast[:,rep],linewidth=0.5,color='red')
            plt.plot(self.__time,norm_fast[:,rep],linewidth=0.4,color='lightgrey')
            plt.plot(self.__time,norm_fast[:,rep],linewidth=0.6,color='black')
            plt.xlim([Tzac2,Tkon2])
            plt.ylim([-0.05,1.05])
            plt.rc('font', size=6)
            plt.xlabel('time (s)', fontsize=6)
            #plt.savefig("binarized/bin_trace_%d.jpg"%(rep),dpi=200,bbox_inches = 'tight')
            plt.savefig("{0}/{1}.pdf".format(directory, rep), dpi=200, bbox_inches='tight')
            plt.close()

    def compare_slow_fast(self):
        properties = {i:{"spikes": 0, "lengths": 0} for i in range(13)}

        for cell in range(self.__number_of_cells):
            start = 0
            end = 0

            for i in range(self.__data_points-1):
                if self.__binarized_slow[i,cell] != self.__binarized_slow[i+1,cell]:
                    end = i+1

                    angle_index = self.__binarized_slow[i,cell]
                    fast_interval = self.__binarized_fast[start:end,cell]

                    bincnt = np.bincount(fast_interval)
                    number_of_spikes = 0 if len(bincnt)==1 else bincnt[1]

                    properties[angle_index]["spikes"] += number_of_spikes
                    properties[angle_index]["lengths"] += end-start

                    start = end

        densities = {i: properties[i]["spikes"]/properties[i]["lengths"] for i in range(13)}

        return densities

    def compare_correlation_distance(self):
        distances = []
        correlations_slow = []
        correlations_fast = []
        for c1 in range(self.__number_of_cells):
            for c2 in range(c1):
                x1, y1 = self.__positions[c1,0], self.__positions[c1,1]
                x2, y2 = self.__positions[c2,0], self.__positions[c2,1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                correlation_slow = np.corrcoef(self.__filtered_slow[:,c1], self.__filtered_slow[:,c2])[0,1]
                correlation_fast = np.corrcoef(self.__filtered_fast[:,c1], self.__filtered_fast[:,c2])[0,1]
                distances.append(distance)
                correlations_slow.append(correlation_slow)
                correlations_fast.append(correlation_fast)

        plt.scatter(distances, correlations_slow)
        plt.scatter(distances, correlations_fast)
        plt.show()


    def exclude(self):
        if self.__excluded is False:
            raise ValueError("No excluded data!")
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data.")
        if self.__binarized_slow is False or self.__filtered_fast is False:
            raise ValueError("No binarized data.")

        deletions = self.__excluded.size

        self.__signal = np.delete(self.__signal, self.__excluded, axis=1)
        self.__positions = np.delete(self.__positions, self.__excluded, axis=0)
        self.__number_of_cells -= deletions

        self.__filtered_slow = np.delete(self.__filtered_slow, self.__excluded, axis=1)
        self.__filtered_fast = np.delete(self.__filtered_fast, self.__excluded, axis=1)
        self.__binarized_slow = np.delete(self.__binarized_slow, self.__excluded, axis=1)
        self.__binarized_fast = np.delete(self.__binarized_fast, self.__excluded, axis=1)

    def build_network(self):
        if self.__filtered_slow is False or self.__filtered_fast is False:
            raise ValueError("No filtered data.")
        if self.__positions is False:
            raise ValueError("No positions.")
        correlation_matrix_slow = np.eye(self.__number_of_cells)
        correlation_matrix_fast = np.eye(self.__number_of_cells)
        for i in range(self.__number_of_cells):
            for j in range(i):
                correlation = np.corrcoef(self.__filtered_slow[:,i], self.__filtered_slow[:,j])[0,1]
                correlation_matrix_slow[i,j] = correlation
                correlation_matrix_slow[j,i] = correlation
                correlation = np.corrcoef(self.__filtered_fast[:,i], self.__filtered_fast[:,j])[0,1]
                correlation_matrix_fast[i,j] = correlation
                correlation_matrix_fast[j,i] = correlation

        A_slow = np.zeros((self.__number_of_cells, self.__number_of_cells))
        A_fast = np.zeros((self.__number_of_cells, self.__number_of_cells))
        for i in range(self.__number_of_cells):
            for j in range(i):
                if correlation_matrix_slow[i,j]>CORR_SLOW:
                    A_slow[i,j] = 1
                    A_slow[j,i] = 1
                if correlation_matrix_fast[i,j]>CORR_FAST:
                    A_fast[i,j] = 1
                    A_fast[j,i] = 1

        self.__G_slow = nx.from_numpy_matrix(A_slow)
        self.__G_fast = nx.from_numpy_matrix(A_fast)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        nx.draw(self.__G_slow, pos=self.__positions, ax=ax[0], with_labels=True, node_color="pink")
        nx.draw(self.__G_fast, pos=self.__positions, ax=ax[1], with_labels=True, node_color="purple")
        plt.show()



    def dynamics_parameters(self):
        if self.__G_slow is False or self.__G_fast is False:
            raise ValueError("No built network.")
        parameters = [dict() for i in range(self.__number_of_cells)]

        for cell in range(self.__number_of_cells):
            spikes = 0
            state = 0
            active_time = 0
            interspike = 0
            interspike_slow = []
            interspike_fast = []

            # Find average frequency of fast data
            for i in range(700*self.__sampling, 1700*self.__sampling):
                if self.__binarized_fast[i,cell] == 1 and state == 0:
                    spikes += 1
                    state = 1
                    interspike_fast.append(interspike/self.__sampling)
                elif self.__binarized_fast[i,cell] == 0 and state == 1:
                    state = 0
                    interspike = 1
                elif self.__binarized_fast[i,cell] == 0 and state == 0:
                    interspike += 1

            parameters[cell]["Ff"] = spikes/self.__data_points*self.__sampling

            # Find average frequency of slow data
            spikes = 0
            state = 0
            interspike = 0
            for i in range(self.__data_points):
                if self.__binarized_slow[i, cell] == 12 and state == 0:
                    spikes += 1
                    state = 1
                    interspike_slow.append(interspike/self.__sampling)
                elif self.__binarized_slow[i,cell] != 12 and state == 1:
                    state = 0
                    interspike = 1

                elif self.__binarized_slow[i,cell] != 12 and state == 0:
                    interspike += 1
            parameters[cell]["Fs"] = spikes/self.__data_points*self.__sampling

            # Find first peak of fast data
            for i in range(150*self.__sampling, self.__data_points):
                if self.__binarized_fast[i,cell] == 1:
                    parameters[cell]["Tfirst"] = i/self.__sampling
                    break

            # Find last peak of fast data
            for i in range(self.__data_points-1, -1, -1):
                if self.__binarized_fast[i,cell] == 1:
                    parameters[cell]["Tlast"] = i/self.__sampling
                    break


            # Find active time of fast data
            active_time = np.bincount(self.__binarized_fast[:,cell])[1]
            parameters[cell]["Ton"] = active_time/self.__sampling

            interspike_fast = interspike_fast[1:-1]

            # Find interspike means
            parameters[cell]["ISIMs"] = np.mean(interspike_slow)
            parameters[cell]["ISIMf"] = np.mean(interspike_fast)

            # Find interspike variations
            parameters[cell]["ISIVs"] = np.std(interspike_slow)/np.mean(interspike_slow)
            parameters[cell]["ISIVf"] = np.std(interspike_fast)/np.mean(interspike_fast)

        return parameters


    def topology_parameters(self):
        if self.__G_slow is False or self.__G_fast is False:
            raise ValueError("No built network.")
        parameters = [dict() for i in range(self.__number_of_cells)]

        for cell in range(self.__number_of_cells):
            parameters[cell]["NDs"] = self.__G_slow.degree[cell]
            parameters[cell]["NDf"] = self.__G_fast.degree[cell]
            parameters[cell]["Cs"] = nx.clustering(self.__G_slow)[cell]
            parameters[cell]["Cf"] = nx.clustering(self.__G_fast)[cell]
            parameters[cell]["NNDs"] = nx.average_neighbor_degree(self.__G_slow)[cell]
            parameters[cell]["NNDf"] = nx.average_neighbor_degree(self.__G_fast)[cell]

        return parameters

    def correlations(self, path):
        dynamics = self.dynamics_parameters()
        topology = self.topology_parameters()
        merged = [{**dynamics[cell], **topology[cell]} for cell in range(self.__number_of_cells)]

        with open(path, "w") as f:
            f.write("p1/p2\t")
            for p2 in merged[0]:
                f.write("{}\t".format(p2))
            f.write("\n")
            for p1 in merged[0]:
                f.write("{}\t".format(p1))
                for p2 in merged[0]:
                    x = [merged[cell][p1] for cell in range(self.__number_of_cells)]
                    y = [merged[cell][p2] for cell in range(self.__number_of_cells)]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    f.write("{}\t".format(round(r_value, 2)))
                f.write("\n")



    # def animation(self):
    #     color = lambda potential: "yellow" if potential==1 else "green"
    #     self.__color_map = np.vectorize(color)(self.__binarized_fast)
    #
    #     fig, ax = plt.subplots()
    #     scatter = nx.draw_networkx_nodes(self.__G,
    #                                      pos=self.__positions,
    #                                      ax=ax,
    #                                      node_color=self.__color_map[0,:])
    #     labels = nx.draw_networkx_labels(self.__G, pos=self.__positions, ax=ax)
    #     lines = nx.draw_networkx_edges(self.__G, pos=self.__positions, ax=ax)
    #
    #     def animate(i, scatter):
    #         if i%1000==0:
    #             print("{0}/{1}".format(i//1000, self.__data_points//1000))
    #         elif i==self.__data_points-1:
    #             print("The end")
    #         colors = self.__color_map[i,:]
    #
    #         try:
    #             rgba_colors = np.array([colorConverter.to_rgba(colors)])
    #         except ValueError:
    #             rgba_colors = np.array([colorConverter.to_rgba(color) for color in colors])
    #         scatter.set_color(rgba_colors)
    #
    #
    #     ani = animation.FuncAnimation(fig, animate,
    #                     frames=self.__data_points, fargs=(scatter, ), interval=1, repeat=False)
    #     ani.save('animation.gif', writer='imagemagick', fps=5)
    #
    #     # plt.show()
