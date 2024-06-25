#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from math import log, sqrt
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import random
import sys
from functools import reduce

sys.setrecursionlimit(1500)

root = Path(__file__).parent.parent
#main_folder = root/'Lab3'
input_folder = root/'resources'
file_input = input_folder/'nodes.json'

# Settings for the plots
csfont = {'fontname': 'Times New Roman', 'fontsize': '13', 'fontstyle': 'normal'}

class Signal_Information:
    def __init__(self, signal_power, path):  # constructor
        self.signal_power = signal_power  # float
        self.noise_power = 0  # float
        self.latency = 0  # float
        self.path = path  # list[string]

    def __repr__(self):
        return f'Nodes(signal_power={self.signal_power}, noise_power={self.noise_power},\
                                     latency={self.latency}, path ={self.path}'

    def update_noise(self, noise):
        self.noise_power = self.noise_power + noise
        print("{:.4g}".format(self.noise_power), 'Watts')

    def update_latency(self, latency):
        self.latency = self.latency + latency
        print("{:.4f}".format(self.latency), 'ms')

    def update_path(self, path):
        path.remove(path[0])  # remove the first element of the list and update path
        self.path = path
        return self.path


class Nodes:
    def __init__(self, label, position, connected_nodes):
        self.label = label  # string (attributes)
        self.position = position  # tuple (float,float)
        self.connected_nodes = connected_nodes  # list[string]
        self.successive = {}  # dictionary[line]

    def __repr__(self):
        return f' Nodes(label={self.label}, position={self.position},\
         connected_nodes={self.connected_nodes}, successive ={self.successive})'

    def propagate_method(self, node, signal_information):  # obj of class signal_info and obj node
        if len(signal_information.path) <= 1:              # Condition for only the last node of the path
            print('Propagating:', node.label)
            signal_information.update_latency(signal_information.latency)
            signal_information.update_noise(signal_information.noise_power)
            print('Final Node, propagate ends')

        if len(signal_information.path) > 1:  # for all other nodes propagate their successive lines
            print('Propagating:', node.label)
            signal_information.update_latency(
                signal_information.latency)  # updating latency information from node instance
            signal_information.update_noise(signal_information.noise_power)  # updating noise power
            for line in node.successive:
                if node.successive[line].label[1] == signal_information.path[1]:
                    # compare if the second letter of the label == second element of the list line:AB  \
                    # [B]==B  path={A,B,D}
                    # Updating path
                    signal_information.update_path(signal_information.path)
                    # propagation lines start using the propagate method
                    node.successive[line].propagate_method(node.successive[line], signal_information)
                    break


class Line:
    def __init__(self, label, length):
        self.label = label  # string
        self.length = length  # float
        self.successive = {}  # dict[node]
        self.state = 'free'  # Binary value 0=Free or 1=Occupied

    def __repr__(self):
        return f'Line( label={self.label}, length={self.length}, successive ={self.successive}) '

    def latency_generation(self, length):
        n = 1  # Glass:1.52 index of refraction
        c = 3 * (10 ** 8)  # Speed of light 3.10^8
        return (length / (2 / 3 * c)) * n

    def noise_generation(self, signal_power, length):
        return 10 ** (-9) * signal_power * length

    def propagate_method(self, line, obj_signal):
        print('Propagating Line', line.label)
        # Generating latency
        latency = line.latency_generation(line.length)
        # Generating noise
        noise = line.noise_generation(obj_signal.signal_power, line.length)
        # updating values in signal_info class
        obj_signal.update_latency(latency)
        obj_signal.update_noise(noise)

        for lin in line.successive:
            if lin == line.label[1]:  # line.successive is a instance Node
                line.successive[lin].propagate_method(line.successive[lin], obj_signal)
            break


class Network:
    def __init__(self, file):  # received instance of calls line and node
        self.nodes = {}  # dictionary for nodes
        self.lines = {}  # dictionary for lines
        self.weighted_paths = pd.DataFrame(columns=['Paths', 'Accumulated Latency', 'Accumulated Noise', 'Accumulated SNR'])

        with open(file, 'r') as f:
            data = json.load(f)
            self.nodes = {i: Nodes(str(i), data[i]['position'], data[i]['connected_nodes']) for i in data}
            # for i in data:   #This code does the same as the line before
            # self.nodes[i] = Nodes(str(i), data[i]['position'], data[i]['connected_nodes'])
            # assigning for each key ('A','B','C'...) an Node instance with
            # all their attribute (label, position, connected nodes)

            for node in self.nodes:  # assigning to each line their successive nodes
                for j in range(len(self.nodes[node].connected_nodes)):
                    # Obtaining the position of Node connected with A, in the first  case B,C,D (Euclidean distance)
                    length = sqrt(
                        pow(self.nodes[node].position[0] - self.nodes[self.nodes[node].connected_nodes[j]].position[0], 2) + \
                        pow(self.nodes[node].position[1] - self.nodes[self.nodes[node].connected_nodes[j]].position[0], 2))
                    # Adding to the Line dictionary
                    self.lines[node + self.nodes[node].connected_nodes[j]] = Line(node + self.nodes[node].connected_nodes[j], length)

    def __repr__(self):
        return f'{self.lines},\n {self.nodes})'

    def connect(self):
        # Method to assign the successive element for each instance Node and Line
        for co_node in self.nodes:
            for line in self.lines:
                if line.startswith(co_node):

                    for i_str in range(len(line)):  # to avoid adding root node in the successive node dictionary
                        if i_str != 0:
                            self.nodes[co_node].successive[line] = self.lines[
                                line]  # filling up the successive dict with the Lines obj
                            self.lines[line].successive[line[i_str]] = self.nodes[line[i_str]]  # 'AB'.successive = 'B'

    def find_path(self, label1, label2, lista):  # start_node:label1 #end_node:label2, empty dict
        list_paths = lista + [label1]  # list of all possible paths for label1
        if label1 == label2:
            return [list_paths]
        if label1 not in self.nodes:  # Check if the label is defined in nodes
            return []
        paths = []  # List of list with paths
        for node in self.nodes[label1].connected_nodes:
            if node not in list_paths:
                # If node is not in the list append the node in the list and search the next one recursively
                new_path = self.find_path(node, label2, list_paths)
                for path in new_path:
                    paths.append(path)
        return paths

    def propagate(self, signal_information):  # receive an object of type signal_info
        for node in self.nodes:
            aux_path = signal_information.path
            if node == aux_path[0]:  # if nro_node is equal to the first element of the list
                print('----------', 'Propagation Starts', '----------------')
                self.nodes[node].propagate_method(self.nodes[node], signal_information)
                break
        return signal_information

    def draw(self):
        # Method to draw the Network
        # It receives a instance Node
        for index_3 in self.nodes:
            coordinate_1 = {  # Dictionary for coordinates of node 2
                'x': self.nodes[index_3].position[0],  # Assigning coord to each point (X,Y)
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
                    plt.text(coordinate_1['x'] - 0.60, coordinate_1['y'] + 0.75, self.nodes[index_3].label,**csfont)
                    plt.text(coordinate_2['x'] - 0.60, coordinate_2['y'] - 0.75, self.nodes[index_4].label, **csfont)
                    plt.xticks(fontsize=10, fontname='Times New Roman')
                    plt.yticks(fontsize=10, fontname='Times New Roman')
                    plt.xlabel('coordinate x', **csfont)
                    plt.ylabel('coordinate y', **csfont)
                    plt.title('Network Topology', **csfont)
                    plt.tight_layout()  # Keep all the elements of the graph tight
        plt.show()

    def dataframe(self, signal_power):
        # Auxiliary lists
        path_df = []
        accumulated_latency = []
        accumulated_noise = []
        accumulated_snr = []

        for node_source in self.nodes:  # node source
            for node_destination in self.nodes:  # node destination
                if node_source != node_destination:  # Finding all possible paths starting from A-B
                    paths_list = self.find_path(self.nodes[node_source].label, self.nodes[node_destination].label, [])
                    for index1 in range(len(paths_list)):  # Iterate over each list
                        path_df.append(reduce(lambda a, b: a + "->" + str(b),
                                              paths_list[index1]))  # Appending each path in the list
                        # Creating a signal_information object for each path
                        signal_info = Signal_Information(signal_power, paths_list[index1])
                        # Propagating signal infor through the path
                        signal_info = self.propagate(signal_info)
                        if signal_info.noise_power != 0:
                            snr = 10 * log(signal_power / signal_info.noise_power)  # Convert to dB
                            accumulated_latency.append(signal_info.latency)  # Appending  updated latency for each path
                            accumulated_noise.append(signal_info.noise_power)# Appending updated noise_pwr for each path
                            accumulated_snr.append(snr)                      # Appending  updated snr for each path

        self.weighted_paths['Paths'] = path_df  # Adding to path to Dataframe
        # Adding lists of accu_latency, accu_noise and SNR values for each path in the df
        self.weighted_paths['Accumulated Latency'] = accumulated_latency
        self.weighted_paths['Accumulated Noise'] = accumulated_noise
        self.weighted_paths['Accumulated SNR'] = accumulated_snr
        # Permanently changes the pandas settings
        pd.set_option('display.max_rows', None)  # Show all the row of the dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        # Convert the whole dataframe as a string and display
        csv = self.weighted_paths.to_csv('weighted_paths.csv')
        print(self.weighted_paths.to_string())

    def find_best_snr(self, node_input, node_output):  # Returns a path with best snr
        # Getting all the possible paths between node1 and node 2
        filter = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node_input) & \
                                     self.weighted_paths['Paths'].str.endswith(node_output)]

        # Searching in the Accumulated SNR column the max value
        # df_value = filter[filter['Accumulated SNR'] == filter['Accumulated SNR'].max()]
        # best_path_1 = df_value['Paths'].values
        path_used = False
        max_snr = (-5000)
        best_path = None
        lines_best_path = list()

        for index, row in filter.iterrows():

            lines_current_list = list()
            resulting_path = row['Paths'].split('->')
            # creating the list of busy paths
            for i in range(len(resulting_path)):
                last_elem = resulting_path[len(resulting_path) - 1]
                first_elem = resulting_path[i]
                if first_elem != last_elem:
                    line = resulting_path[i] + resulting_path[i + 1]
                    lines_current_list.append(line)

            path_used = False
            # Verifying if the path is already occupied
            for line in lines_current_list:
                if self.lines[line].state == 'occupied':
                    path_used = True

            current_snr = row['Accumulated SNR']
            # if first_cycle == True:
            #     first_cycle = False
            #     max_snr = current_snr
            #     best_path = row['Paths']
            #     lines_previous_path = lines_current_list
            #     for line in lines_current_list:
            #         if self.lines[line].state == 'free':
            #             self.lines[line].state = "occupied"

            if (current_snr > max_snr) and (path_used is False):
                max_snr = current_snr
                best_path = row['Paths']
                lines_best_path = lines_current_list

        return lines_best_path, best_path
        # for idx in lines_previous_path:
        #     self.lines[idx].state = 'free'
        # lines_previous_path = lines_current_list
        # for line in lines_current_list:
        #     self.lines[line].state = "occupied"


    def find_best_latency(self, node1, node2): # Returns a path with best latency (min latency)
        # Filtering the df Path column, to find the required path
        filter_path = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node1) &
                                          self.weighted_paths['Paths'].str.endswith(node2)]
        # Searching in the Accumulated Latency column the min value
        # value_latency = filter_path[filter_path['Accumulated Latency'] == filter_path['Accumulated Latency'].min()]
        path_used = False
        max_latency = float('-inf')
        best_path = None
        lines_best_path = list()

        for index, row in filter_path.iterrows():
            lines_current_list = list()
            resulting_path = row['Paths'].split('->')

            # creating the list of busy paths
            for i in range(len(resulting_path)):
                last_elem = resulting_path[len(resulting_path) - 1]
                first_elem = resulting_path[i]
                if first_elem != last_elem:
                    line = resulting_path[i] + resulting_path[i + 1]
                    lines_current_list.append(line)

            path_used = False
            # Verifying if the path is already occupied
            for line in lines_current_list:
                if self.lines[line].state == 'occupied':
                    path_used = True

            current_latency = row['Accumulated Latency']
            if current_latency is not None and current_latency > max_latency and (path_used is False):
                max_latency = current_latency
                best_path = row['Paths']
                lines_best_path = lines_current_list
        return lines_best_path, best_path

    def stream(self, connection_list, signal_power, label):

        for list in range(len(connection_list)):
            if (label == 'latency') or (label == 'Latency'):
                lines, best_path = self.find_best_latency(connection_list[list].input, connection_list[list].output)
            else:  # if (label == 'snr') or (label == 'SNR'):
                # For the SNR
                lines, best_path = self.find_best_snr(connection_list[list].input, connection_list[list].output)

            if best_path is not None:
                # Converting best_latency_path array in a string
                resulting_path = best_path.split('->')             # ['A'->'C'->'E']-->['A','C','E']

                # Propagating through the path found before
                signal_propagation = Signal_Information(signal_power, resulting_path)
                propagation = self.propagate(signal_propagation)

                # Modifying the latency and snr of the path between the two nodes
                connection_list[list].latency = propagation.latency
                if propagation.noise_power != 0:
                    connection_list[list].snr = 10 * log(signal_power / propagation.noise_power)  # In dB
                # Updating lines states
                for line in lines:
                    if self.lines[line].state == 'free':            # if the line is 'Free'=1 then change the state
                        self.lines[line].state = "occupied"         # free=1 or occupied=0
            else:
                # When there is no path available case
                connection_list[list].latency = 0
                connection_list[list].snr = None                    # SNR [dB]

class Connection:
    def __init__(self, input_node, output_node, signal_power):
        self.input = input_node  # string
        self.output = output_node  # string
        self.signal_power = signal_power  # float
        self.latency = 0.0  # float
        self.snr = 0.0  # float

    def __repr__(self):
        return f'{self.input}, {self.output}, {self.signal_power}, {self.latency}, {self.snr})'

class Lightpath:
    def __init__(self, signal_power, path, index):  # constructor
        self.signal_power = signal_power  # float
        self.noise_power = 0              # float
        self.latency = 0                  # float
        self.path = path                  # list[string]
        self.channel = index # index or each channel

def main():
    signal_power = 1 * 10 ** (-3)
    # Initialization of the Network
    net = Network(file_input)
    net.connect()
    # net.draw()
    net.dataframe(signal_power)

    # Creating 100 Connections
    lista_connection = []
    for i in range(100):
        node1 = random.choice(list(net.nodes.keys()))
        node2 = random.choice(tuple(net.nodes.keys() - {node1}))
        if node1 != node2:
            lista_connection.append(Connection(node1, node2, signal_power))  # Connections lists

    # Stream call
    net.stream(lista_connection, signal_power, 'SNR')

    # Selecting best snr
    best_snr = []
    path_snr = []
    print('Best SNR found')
    for i in range(len(lista_connection)):
        print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
              lista_connection[i].snr)
        best_snr.append(lista_connection[i].snr)
        path_snr.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Selecting best
    best_latency = []
    path_latency = []
    print('Best latency found')
    for i in range(len(lista_connection)):
        print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
              lista_connection[i].latency)
        best_latency.append(lista_connection[i].latency)
        path_latency.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Plotting the snr and latency distributions
    snr_array = [0 if value is None else value for value in best_snr]
    latency_array = np.array(best_latency)

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    # For SNRs
    axs[0].hist(snr_array, color='maroon', histtype='stepfilled', bins=30, edgecolor='k', density=True)
    axs[0].set_xlabel('SNR dB', **csfont)
    axs[0].set_ylabel('Number of Connections', **csfont)
    axs[0].set_title('SNR for 100 Connections', **csfont)
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')

    # For latencies
    axs[1].hist(latency_array, color='#ff7f0e', bins=30, histtype='stepfilled', edgecolor='k', density=True)
    axs[1].set_title('Latencies for 100 Connections', **csfont)
    axs[1].set_xlabel('Latency sec', **csfont)
    axs[1].set_ylabel('Number of Connections', **csfont)
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')

    plt.show()
    plt.tight_layout()

if __name__ == '__main__':
    main()
