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
        plt.xlim(self.settings["binarize"]["image"])
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
        plt.xlim(self.settings["binarize"]["image"])
        plt.ylim([-0.05,1.05])
        plt.rc('font', size=6)
        plt.xlabel('time (s)', fontsize=6)
        #plt.savefig("binarized/bin_trace_%d.jpg"%(rep),dpi=200,bbox_inches = 'tight')
        plt.savefig("{0}/{1}.png".format(directory, rep), dpi=200, bbox_inches='tight')
        plt.close()
