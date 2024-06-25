# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from math import log, sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import random
import sys
from functools import reduce
from parameters import *
# Allows to show all the lines of the Dataframes
sys.setrecursionlimit(1500)

root = Path(__file__).parent.parent
# main_folder = root/'Lab3'
input_folder = root/'resources'
file_input = input_folder/'nodes.json'

# Settings for the plots
csfont = {'fontname': 'Times New Roman', 'fontsize': '13', 'fontstyle': 'normal'}


class Signal_Information(object):

    def __init__(self, signal_power=None, path=None):  # constructor
        if signal_power:
            self._signal_power = signal_power   # float
        else:
            self._signal_power = 0
        self._noise_power = 0                   # float
        self._latency = 0                       # float
        if path:
            self._path = path                   # list[string]
        else:
            self._path = []

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        """Latency in seconds"""
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    def __repr__(self):
        return f'Nodes(signal_power={self.signal_power}, noise_power={self.noise_power},\
                                     latency={self.latency}, path ={self.path}'

    def update_noise(self, noise):
        self.noise_power = self.noise_power + noise
        print("{:.4f}".format(self.noise_power), 'Watts')

    def update_latency(self, latency):
        self.latency = self.latency + latency
        print("{:.4f}".format(self.latency), 'ms')

    def update_path(self, path):
        path.remove(path[0])        # remove the first element of the list and update path
        self.path = path
        return self.path


class Lightpath(Signal_Information):
    # Lightpath inherits all the methods and attributes of Signal Information Class
    def __init__(self, signal_power, path, channel):    # constructor of Lightpath
        super().__init__(signal_power, path)         # Initializing constructor of Signal Information
        self.channel = channel                          # index of the channel


class Nodes(object):

    def __init__(self, label, position, connected_nodes):
        self.label = label                          # string (attributes)
        self.position = position                    # tuple (float,float)
        self.connected_nodes = connected_nodes      # list[string]
        self.successive = {}                        # dictionary[line]
        self.switching_matrix = None

    def __repr__(self):
        return f' Nodes(label={self.label}, position={self.position},\
         connected_nodes={self.connected_nodes}, successive ={self.successive})'

    def propagate_method(self, node, signal_information):  # obj of class signal_info and node
        if len(signal_information.path) <= 1:              # Condition for only the last node of the path
            print('Propagating:', node.label)
            signal_information.update_latency(signal_information.latency)
            signal_information.update_noise(signal_information.noise_power)
            print('Final Node, propagate ends')

        if len(signal_information.path) > 1:  # for all other nodes propagate their successive lines
            print('Propagating:', node.label)
            signal_information.update_latency(signal_information.latency)    # updating latency from node instance
            signal_information.update_noise(signal_information.noise_power)  # updating noise power
            for line in node.successive:
                if node.successive[line].label[1] == signal_information.path[1]:
                    # compare if the second letter of the label == second element of the list path:AB  \
                    # [B]==B  path={A,B,D}
                    # Updating path
                    signal_information.update_path(signal_information.path)
                    # propagation lines start using the propagate method
                    node.successive[line].propagate_method(node.successive[line], signal_information)
                    break

class Line(object):

    def __init__(self, label, length):
        self.label = label        # string
        self.length = length      # float
        self.successive = {}      # dict[node]
        self.state = np.ones(NUMBER_OF_CHANNELS, dtype=int)       # free=1 or occupied=0

    def __repr__(self):
        return f'Line(label={self.label}, length={self.length}, successive ={self.successive})'

    def latency_generation(self, length):
        n = 1              # Glass:1.52 index of refraction
        c = 3 * (10 ** 8)  # Speed of light 3.10^8
        return (length / (2/3 * c)) * n

    def noise_generation(self, signal_power, length):
        return 10 ** (-9) * signal_power * length

    def propagate_method(self, line, signal_information):      # Propagates the signal information along the line
        print('Propagating Line', line.label)                  # that signal_info implies the info of the lightpaths too
        # Generating latency
        latency = line.latency_generation(line.length)
        # Generating noise
        noise = line.noise_generation(signal_information.signal_power, line.length)
        # updating values in signal_info class
        signal_information.update_latency(latency)
        signal_information.update_noise(noise)

        for node in line.successive:
            if node == line.label[1]:  # line.successive is a instance Node 'B' == [B]
                line.successive[node].propagate_method(line.successive[node], signal_information)
            break

class Network(object):

    def __init__(self, file):  # received instance of calls line and node
        self.nodes = {}        # dictionary for nodes
        self.lines = {}        # dictionary for lines
        self.weighted_paths = pd.DataFrame(columns=['Paths', 'Accumulated Latency', 'Accumulated Noise',
                                                    'Accumulated SNR'])
        self.route_space = pd.DataFrame(columns=['Paths','1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

        with open(file, 'r') as f:
            data = json.load(f)
            self.nodes = {i: Nodes(str(i), data[i]['position'], data[i]['connected_nodes']) for i in data}

            # Assigning for each key ('A','B','C'...) a Node instance with
            # all its attributes (label, position, connected nodes)

            for node in self.nodes:         # assigning to each line their successive nodes
                for j in range(len(self.nodes[node].connected_nodes)):
                    # Obtaining the position of Node connected with A, in the first  case B,C,D (Euclidean distance)
                    x1=self.nodes[node].position[0]
                    x2=self.nodes[self.nodes[node].connected_nodes[j]].position[0]
                    y1=self.nodes[node].position[1]
                    y2=self.nodes[self.nodes[node].connected_nodes[j]].position[1]
                    length = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
                    # Adding to the Line dictionary
                    self.lines[node + self.nodes[node].connected_nodes[j]] = Line(node + self.nodes[node].connected_nodes[j], length)

    def __repr__(self):
        return f'{self.lines},\n {self.nodes})'

    def connect(self):
        # Method to assign the successive element for each instance Node and Line
        for node in self.nodes:
            for line in self.lines:
                if line.startswith(node):
                    for i in range(len(line)):  # to avoid adding root node in the successive node dictionary
                        if i != 0:
                            # Assigning to the Nodes and Lines the successive elements (nodes and lines)
                            self.nodes[node].successive[line] = self.lines[line]        # 'A'.successive = 'AB'
                            self.lines[line].successive[line[i]] = self.nodes[line[i]]  # 'AB'.successive = 'B'

        # For Switching Matrix
        for node in self.nodes:
            sw2_dict = dict()
            node_successive_list = list()
            for line in self.lines:   # ['AB']
                if line.startswith(node):

                    for node_connected in self.lines[line].successive:
                        node_successive_list.append(self.nodes[node_connected].label)

            for key_node in node_successive_list:
                sw1_dict = dict()
                for k in node_successive_list:
                    if key_node == k:
                        sw1_dict[self.nodes[k].label] = np.zeros(NUMBER_OF_CHANNELS, dtype=int)
                    elif key_node != k:
                        # Initializing the switching_matrix when couple of nodes are different
                        sw1_dict[self.nodes[k].label] = np.ones(NUMBER_OF_CHANNELS, dtype=int)

                sw2_dict[key_node]= sw1_dict                 #{'B':sw1_dict}
            self.nodes[node].switching_matrix = sw2_dict

    def find_path(self, node_input, node_output, lista):
        list_paths = lista + [node_input]     # list of all possible paths for label1
        if node_input == node_output:
            return [list_paths]
        if node_input not in self.nodes:      # Check if the label is defined in nodes
            return []
        paths = []                        # List of list with paths
        for node in self.nodes[node_input].connected_nodes:
            if node not in list_paths:
                # If node is not in the list append the node in the list and search the next one recursively
                new_path = self.find_path(node, node_output, list_paths)
                for path in new_path:
                    paths.append(path)
        return paths

    def propagate(self, lightpath):
        for node in self.nodes:
            aux_path = lightpath.path
            if node == aux_path[0]:                 # if node is equal to the first element of the list path
                print('------------', 'Propagation Starts', '-------------')
                self.nodes[node].propagate_method(self.nodes[node], lightpath)
                break
        return lightpath

    def propagate_probe(self, signal_information):
        for node in self.nodes:
            aux_path = signal_information.path
            if node == aux_path[0]:  # if nro_node is equal to the first element of the list
                print('------------', 'Propagation Starts', '--------------')
                self.nodes[node].propagate_method(self.nodes[node], signal_information)
                break
        return signal_information


    def draw(self):
        # Method to draw the Network
        # It receives a instance Node
        for index_3 in self.nodes:
            coordinate_1 = {  # Dictionary for coordinates of node 2
                'x': self.nodes[index_3].position[0],     # Assigning coord to each point (X,Y)
                'y': self.nodes[index_3].position[1]}
            for index_4 in self.nodes[index_3].connected_nodes:
                coordinate_2 = {  # Dictionary for coordinates of successive node
                    'x': self.nodes[index_4].position[0],
                    'y': self.nodes[index_4].position[1]}  # To avoid create 2 lines connecting the same

                # Comparing successive node with the previous node
                # ex: if we are in node A; B>A plot the line, if C>A plot  line
                if self.nodes[index_4].label > self.nodes[index_3].label:
                    x_values = [coordinate_1['x'], coordinate_2['x']]  # Assigning X1,X2 in a list
                    y_values = [coordinate_1['y'], coordinate_2['y']]  # Assigning Y1,Y2 in a list

                    # Plotting each pair of lines
                    plt.plot(x_values, y_values, '-ok', mfc='C1', mec='C1')
                    # Adding labels to each node
                    plt.text(coordinate_1['x'] - 0.60, coordinate_1['y'] + 0.75, self.nodes[index_3].label, **csfont)
                    plt.text(coordinate_2['x'] - 0.60, coordinate_2['y'] - 0.75, self.nodes[index_4].label, **csfont)
                    plt.xticks(fontsize=10, fontname='Times New Roman')
                    plt.yticks(fontsize=10, fontname='Times New Roman')
                    plt.xlabel('coordinate x', **csfont)
                    plt.ylabel('coordinate y', **csfont)
                    plt.title('Network Topology', **csfont)
                    plt.tight_layout()
        plt.style.use('seaborn-paper')
        plt.show()

    def weighted_paths_dataframe(self, signal_power):
        # Auxiliary lists
        path_df = []
        accumulated_latency = []
        accumulated_noise = []
        accumulated_snr = []

        for node_source in self.nodes:  # node source
            for node_destination in self.nodes:  # node destination
                if node_source != node_destination:  # Finding all possible paths starting from A-B
                    paths_list = self.find_path(self.nodes[node_source].label, self.nodes[node_destination].label, [])
                    for index1 in range(len(paths_list)):
                        path_df.append(reduce(lambda a, b: a + "->" + str(b), paths_list[index1]))
                        # Appending each path in the list

                        # Propagation of the signal along all the possible paths
                        # Gets latency, snr and noise power of all the possible paths
                        # Using Signal Information
                        signal_info = Signal_Information(signal_power, paths_list[index1])
                        signal_info= self.propagate_probe(signal_info)      # probe method = old propagate method
                                                                            # but using the signal information
                        if signal_info.noise_power != 0:
                            snr = 10 * np.log10(signal_info.signal_power / signal_info.noise_power)  # Convert to dB
                            accumulated_latency.append(signal_info.latency)   # Appending updated latency for each path
                            accumulated_noise.append(signal_info.noise_power) # Appending updated noise power for e/path
                            accumulated_snr.append(snr)                       # Appending updated snr for each path
                        else:
                            print('Error in the noise power')
        # Adding lists of paths, accu_latency, accu_noise and SNR values for each path in the df
        self.weighted_paths['Paths'] = path_df
        self.weighted_paths['Accumulated Latency'] = accumulated_latency
        self.weighted_paths['Accumulated Noise'] = accumulated_noise
        self.weighted_paths['Accumulated SNR'] = accumulated_snr
        # For Route_Space DF
        self.route_space['Paths'] = path_df
        # Set all values in columns '1' to '10' to 1 (Initializing DF)
        self.route_space[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']] = 1       # free=1 or occupied=0

        # Permanently changes the pandas settings
        pd.set_option('display.max_rows', None)      # Show all the rows of the dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        #print(self.route_space.to_string())
        print(self.weighted_paths.to_string())

    # Returns a path with best snr
    def find_best_snr(self, node_input, node_output):
        # Getting all the possible paths between node1 and node 2
        paths_btw_nodes_df = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node_input) &
                                                 self.weighted_paths['Paths'].str.endswith(node_output)]

        max_snr = (-5000)
        best_path = 'NaN'
        best_path_index = -1
        index_channel = -1

        for index, row in paths_btw_nodes_df.iterrows():
            for channel in range(1,NUMBER_OF_CHANNELS+1):

                current_snr = row['Accumulated SNR']
                idx = [idx for idx, col in enumerate(self.route_space.columns) if str(channel) == col]
                column_channel = str(idx[0])
                if (current_snr > max_snr) and (self.route_space.loc[index, column_channel] == 1):
                    max_snr = current_snr
                    best_path = row['Paths']
                    best_path_index = index
                    index_channel = channel

        if index_channel is not -1:

            lines_current_list = list()
            resulting_path = best_path.split('->')
            # creating the list of paths
            for i in range(len(resulting_path)):
                last_elem = resulting_path[len(resulting_path) - 1]
                first_elem = resulting_path[i]
                if first_elem != last_elem:
                    line = resulting_path[i] + resulting_path[i + 1]
                    lines_current_list.append(line)
            # Changing the current state of the channel throughout the lines of the path
            for line in lines_current_list:
                self.lines[line].state[index_channel-1] = 0       # free=1 or occupied=0

        return best_path_index, index_channel

    def find_best_latency(self, node_input, node_output):
        # Returns the path with the best latency (min latency)
        # Filtering the df Path column, to find the required path
        filter_path = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node_input) &
                                          self.weighted_paths['Paths'].str.endswith(node_output)]

        max_latency = float('-inf')
        best_path = 'NaN'
        best_path_index = -1
        index_channel = -1

        for index, row in filter_path.iterrows():
            for channel in range(1,NUMBER_OF_CHANNELS+1):

                current_latency = row['Accumulated Latency']
                idx = [idx for idx, col in enumerate(self.route_space.columns) if str(channel) == col]
                column_channel = str(idx[0])
                if (current_latency > max_latency) and (self.route_space.loc[index, column_channel] == 1):
                    max_latency = current_latency
                    best_path = row['Paths']
                    best_path_index = index
                    index_channel = channel

        if index_channel is not -1:

            lines_current_list = list()
            resulting_path = best_path.split('->')
            # Creating the list of busy paths
            for i in range(len(resulting_path)):
                last_elem = resulting_path[len(resulting_path) - 1]
                first_elem = resulting_path[i]
                if first_elem != last_elem:
                    line = resulting_path[i] + resulting_path[i + 1]
                    lines_current_list.append(line)
            # Changing the current state of the channel throughout the lines of the path
            for line in lines_current_list:
                self.lines[line].state[index_channel-1] = 0                 # free=1 or occupied=0

        return best_path_index, index_channel

    def route_space_update(self,):
        # Updating all paths in the route_space performing for each path the multiplication of all the state line
        # arrays and all switching matrix array that compose the path (excluding the switching matrix of the
        # initial and last nodes).
        paths = self.weighted_paths['Paths']
        index = 0
        for path in paths:
            channel = np.ones(NUMBER_OF_CHANNELS, dtype=int)
            path_str = str(path).split('->')
            for i in range(len(path_str)):
                if i != 0 and (i != len(path_str)-1) and i % 2 == 0:
                    channel *= self.lines[path_str[i-1] + path_str[i]].state
                    channel *= self.nodes[path_str[i]].switching_matrix[path_str[i-1]][path_str[i+1]]
                elif (i != len(path_str)-1) and i % 2 != 0:
                    channel *= self.lines[path_str[i-1] + path_str[i]].state
                    channel *= self.nodes[path_str[i]].switching_matrix[path_str[i - 1]][path_str[i + 1]]
                elif i != 0:
                    channel *= self.lines[path_str[i-1] + path_str[i]].state

            # Updating channel availability of Route Space DF
            self.route_space.iloc[index, 1:] = channel
            index += 1


    def stream(self, connection_list, signal_power, label):

        for connection in range(len(connection_list)):
            if (label == 'latency') or (label == 'Latency'):
                best_path_index, index_channel = self.find_best_latency(connection_list[connection].input,
                                                                        connection_list[connection].output)
            else:  # if (label == 'snr') or (label == 'SNR'):
                # For the SNR
                best_path_index, index_channel = self.find_best_snr(connection_list[connection].input,
                                                                    connection_list[connection].output)

            if best_path_index != -1:
                best_path = self.weighted_paths.loc[best_path_index,'Paths']
                resulting_path = str(best_path).split('->')      # ['A'->'C'->'E']->['A','C','E']
                # Propagating throughout the lightpath the signal information
                lightpath = Lightpath(signal_power, resulting_path, index_channel)
                self.propagate(lightpath)
                connection_list[connection].latency = lightpath.latency
                connection_list[connection].snr = 10 * np.log10(lightpath.signal_power/lightpath.noise_power) # SNR [dB]
                # Updating the rout_space dataframe by searching using the index path and the index_channel=column
                if best_path_index != -1 and index_channel != -1:
                    self.route_space_update()
            else:
                # When there is no path available case
                connection_list[connection].latency = 0
                connection_list[connection].snr = None    # In dB
        print(self.route_space.to_string())


class Connection(object):

    def __init__(self, input_node, output_node, signal_power):
        self.input = input_node             # string
        self.output = output_node           # string
        self.signal_power = signal_power    # float
        self.latency = 0.0                  # float
        self.snr = 0.0                      # float

    def __repr__(self):
        return f'{self.input}, {self.output}, {self.signal_power}, {self.latency}, {self.snr})'

def main():
    signal_power = 1e-3
    # Initialization of the Network
    net = Network(file_input)
    net.connect()
    # net.draw()
    net.weighted_paths_dataframe(signal_power)

    # Creating 100 Connections
    lista_connection = []
    for i in range(100):
        node1 = random.choice(list(net.nodes.keys()))
        node2 = random.choice(tuple(net.nodes.keys() - {node1}))
        if node1 != node2:
            lista_connection.append(Connection(node1, node2, signal_power))

    # Stream call
    label = 'latency'
    net.stream(lista_connection, signal_power, label)

    # Selecting best snr
    best_snr = []
    path_snr = []
    print('\n''\n''Best SNR found')
    for i in range(len(lista_connection)):
        print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
              lista_connection[i].snr)
        best_snr.append(lista_connection[i].snr)
        path_snr.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Selecting best
    best_latency = []
    path_latency = []
    print('\n''\n''Best latency found')
    for i in range(len(lista_connection)):
        print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
              lista_connection[i].latency)
        best_latency.append(lista_connection[i].latency)
        path_latency.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Plotting the snr and latency distributions
    snr_array = [0 if value is None else value for value in best_snr]
    snr_array = np.array(snr_array)
    latency_array = np.array(best_latency)

    plt.figure(figsize=(8, 5))
    if label == 'SNR' or label=='snr':
        # For SNRs
        #plt.hist(snr_array, color='maroon', histtype='stepfilled', bins=30, edgecolor='k')
        sns.histplot(snr_array, color='red', kde=False, bins=30, edgecolor='k', fill=True)
        plt.xlabel('SNR [dB]', **csfont)
        plt.ylabel('Number of Connections', **csfont)
        plt.title('SNR for 100 Connections', **csfont)
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
    else:
        # For latencies
        #plt.hist(latency_array*1e3, color='#ff7f0e', bins=30, histtype='stepfilled', edgecolor='k')
        sns.histplot(latency_array, color='red', kde=False, bins=30, edgecolor='k', fill=True)
        plt.title('Latencies for 100 Connections', **csfont)
        plt.xlabel('Latency [s]', **csfont)
        plt.ylabel('Number of Connections', **csfont)
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')

    plt.style.use('seaborn-paper')
    plt.show()
    plt.tight_layout()

if __name__ == '__main__':
    main()
