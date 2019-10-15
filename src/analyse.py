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


    x0 = (np.pi/3 - np.pi/6)/2
    d = np.pi/6
    phi = [x0+d*i for i in range(12)]
    bars = np.array([(-np.cos(i)+1)/2 for i in phi])
    data = np.array([densities[i] for i in range(1, len(densities))])
    norm_data = data/np.max(data)

    # plt.plot(phi, bars, phi, norm_data)
    # plt.show()

    # colors = plt.cm.seismic(norm_data)
    # ax = plt.subplot(111, projection="polar")
    # ax.bar(phi, norm_data, width=2*np.pi/12, bottom=0.0, color=colors)
    # plt.show()

    return (phi, bars, norm_data)

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

    return (distances, correlations_slow, correlations_fast)

    # plt.scatter(distances, correlations_slow)
    # plt.scatter(distances, correlations_fast)
    # plt.show()


    # slow_intervals = {key: 0 for key in range(0, int(max(distances)),20)}
    # fast_intervals = {key: 0 for key in range(0, int(max(distances)),20)}
    # for start in slow_intervals:
    #     for i in range(len(distances)):
    #         if distances[i] >= start and distances[i] < start+20:
    #             slow_intervals[start] += correlations_slow[i]
    #             fast_intervals[start] += correlations_fast[i]
    # return (slow_intervals, fast_intervals)




def get_positions(self):
    return self.__positions


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

def build_network(self, slow_threshold, fast_threshold):
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
            if correlation_matrix_slow[i,j]>self.settings["slow_threshold"]:
                A_slow[i,j] = 1
                A_slow[j,i] = 1
            if correlation_matrix_fast[i,j]>self.settings["fast_threshold"]:
                A_fast[i,j] = 1
                A_fast[j,i] = 1

    self.__G_slow = nx.from_numpy_matrix(A_slow)
    self.__G_fast = nx.from_numpy_matrix(A_fast)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    nx.draw(self.__G_slow, pos=self.__positions, ax=ax[0], with_labels=True, node_color="pink")
    nx.draw(self.__G_fast, pos=self.__positions, ax=ax[1], with_labels=True, node_color="purple")
    plt.show()

    average_slow_degree = np.mean([self.__G_slow.degree[i] for i in self.__G_slow])
    average_fast_degree = np.mean([self.__G_fast.degree[i] for i in self.__G_fast])

    # return (average_slow_degree, average_fast_degree)
    return (average_slow_degree, average_fast_degree)

def get_adjacency(self):
    adjacency_slow = nx.to_numpy_matrix(self.__G_slow)
    adjacency_fast = nx.to_numpy_matrix(self.__G_fast)
    return (adjacency_slow, adjacency_fast)

def get_excluded_positions(self, path):
    np.savetxt(path, self.__positions)



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
        start = int(Tzac3*self.__sampling)
        stop = int(Tkon3*self.__sampling)
        for i in range(start, stop):
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
        i_start = 0
        i_end = 0
        end_check = 0
        for i in range(self.__data_points):
            if self.__binarized_slow[i, cell] == 12 and state == 0:
                if i_start == 0:
                    i_start = i
                spikes += 1
                if i_end==0 or end_check!=spikes:
                    i_end = i
                    end_check = spikes
                state = 1
                interspike_slow.append(interspike/self.__sampling)
            elif self.__binarized_slow[i,cell] != 12 and state == 1:
                state = 0
                interspike = 1

            elif self.__binarized_slow[i,cell] != 12 and state == 0:
                interspike += 1
        parameters[cell]["Fs"] = (spikes-1)/(i_end-i_start)*self.__sampling

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
