#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from math import log,sqrt
import sys
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from pathlib import Path

sys.setrecursionlimit(1500)
# Exercise Lab3: Network

root = Path(__file__).parent.parent
# main_folder = root/'Lab3'
input_folder = root/'resources'
file_input = input_folder/'nodes.json'

# Load the Network from the JSON file, connect nodes and lines in Network.
# Then propagate a Signal Information object of 1mW in the network and save the results in a dataframe.
# Convert this dataframe in a csv file called 'weighted_path' and finally plot the network.
# Follow all the instructions in README.md file
# Settings for the plots
csfont = {'fontname': 'Times New Roman', 'fontsize': '13', 'fontstyle': 'normal'}

class Signal_Information:
    def __init__(self, signal_power, path):  # constructor
        self.signal_power = signal_power     # float
        self.noise_power = 0                 # float
        self.latency = 0                     # float
        self.path = path                     # list[string]

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
        self.label = label                      # string (attributes)
        self.position = position                # tuple (float,float)
        self.connected_nodes = connected_nodes  # list[string]
        self.successive = {}                    # dictionary[line]

    def __repr__(self):
        return f' Nodes(label={self.label}, position={self.position}, connected_nodes={self.connected_nodes},' \
               f'successive ={self.successive}'

    def propagate_method(self, node, signal_information):                # obj of class signal_info
        if len(signal_information.path) <= 1:                            # Condition for only the last node of the path
            print('Propagating:', node.label)
            signal_information.update_latency(signal_information.latency)
            signal_information.update_noise(signal_information.noise_power)
            print('Final Node, propagate ends')

        if len(signal_information.path) > 1:                                  # for all other nodes propagate their successive lines
            print('Propagating:', node.label)
            signal_information.update_latency(signal_information.latency)     # upadting latency information from node instance
            signal_information.update_noise(signal_information.noise_power)   # updating noise power
            for line in node.successive:
                # compare if the second letter of the label == second element of the list, line:AB [B]==B path={A,B,D}
                if node.successive[line].label[1] == signal_information.path[1]:
                    signal_information.update_path(signal_information.path)      # Updating path
                    node.successive[line].propagate_method(node.successive[line], signal_information)    # propagation lines start
                    break

class Line:
    def __init__(self, label, length):
        self.label = label    # string
        self.length = length  # float
        self.successive = {}  # dict[node]

    def __repr__(self):
        return f'Line( label={self.label}, length={self.length}, successive ={self.successive}) '

    def latency_generation(self, length):
        n = 1                # Glass:1.52 index of refraction
        c = 3 * 10 ^ 8       # Speed of light
        return (length / ((2/3)*c) ) * n

    def noise_generation(self, signal_power, length):
        return 10 ** (-9) * signal_power * length

    def propagate_method(self, line, obj_signal):           # received Line instance,signal object instance, length
        print('Propagating Line', line.label)
        latency = line.latency_generation(line.length)
        noise = line.noise_generation(obj_signal.signal_power,line.length)
        obj_signal.update_latency(latency)
        obj_signal.update_noise(noise)

        for lin in line.successive:
            if lin == line.label[1]:                        # line.successive is a instance Node
                line.successive[lin].propagate_method(line.successive[lin], obj_signal)
            break

class Network:
    def __init__(self, file):  # received instance of calls line and node
        self.nodes = {}  # dictionary for nodes
        self.lines = {}  # dictionary for lines
        self.weighted_paths = pd.DataFrame(
            columns=['Paths', 'Accumulated Latency', 'Accumulated Noise', 'Accumulated SNR'])

        with open(file, 'r') as f:
            data = json.load(f)
            self.nodes = {i: Nodes(str(i), data[i]['position'], data[i]['connected_nodes']) for i in data}
            ## for i in data:   #This code does the same as the line before
            ## self.nodes[i] = Nodes(str(i), data[i]['position'], data[i]['connected_nodes'])
            # assigning for each key ('A','B','C'...) an Node instance with
            # all their attribute (label, position, connected nodes)

            for node in self.nodes:  # assigning to each line their successive nodes
                for j in range(len(self.nodes[node].connected_nodes)):
                    # Obtaining the position of Node connected with A, in the first  case B,C,D (Euclidean distance)
                    length = sqrt(
                        pow(self.nodes[node].position[0] - self.nodes[self.nodes[node].connected_nodes[j]].position[0], 2) +\
                        pow(self.nodes[node].position[1] - self.nodes[self.nodes[node].connected_nodes[j]].position[0], 2))
                    # Adding lines to the Line dictionary
                    self.lines[node + self.nodes[node].connected_nodes[j]] = Line(
                        node + self.nodes[node].connected_nodes[j], length)

    def __repr__(self):
        return f'{self.lines},\n {self.nodes})'

    def connect(self):
        # Method to assign the successive element for each instance Node and Line
        for co_node in self.nodes:
            for line in self.lines:
                if line.startswith(co_node):

                    for i_str in range(len(line)):  # to avoid adding root node in the successive node dictionary
                        if i_str != 0:
                            # Assigning to the Nodes and Lines dict, the successive elements (nodes and lines)
                            self.nodes[co_node].successive[line] = self.lines[line]
                            self.lines[line].successive[line[i_str]] = self.nodes[line[i_str]]  # 'AB'.successive = 'B'

    def find_path(self, label1, label2, lista):
        # Method to find all the possible paths between two nodes
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

    def propagate(self, Signal_Information):  # receive an object of type signal_info
        for node in self.nodes:
            aux_path = Signal_Information.path
            if node == aux_path[0]:  # if nro_node is equal to the first element of the list
                print('----------', 'Propagation Starts', '----------------')
                self.nodes[node].propagate_method(self.nodes[node], Signal_Information)
                break
        return Signal_Information

    def draw(self):
        # Method to draw the Network
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
                    plt.text(coordinate_2['x'] - 0.60, coordinate_2['y'] - 0.75, self.nodes[index_4].label,**csfont)
                    plt.xticks(fontsize=10, fontname='Times New Roman')
                    plt.yticks(fontsize=10, fontname='Times New Roman')
                    plt.xlabel('coordinate x', **csfont)
                    plt.ylabel('coordinate y', **csfont)
                    plt.title('Network Topology', **csfont)
                    plt.tight_layout()  # Keep all the elements of a graph tight
        plt.show()
    def dataframe(self, signal_power):
        # Auxiliary lists
        path_df = []
        accumulated_latency = []
        accumulated_noise = []
        accumulated_snr = []

        for node_source in self.nodes:
            for node_destination in self.nodes:
                if node_source != node_destination:             # Finding all possible paths starting from node A
                    paths_list = self.find_path(self.nodes[node_source].label, self.nodes[node_destination].label,
                                                [])
                    for index1 in range(len(paths_list)):       # Appending each path in the list
                        path_df.append(reduce(lambda a, b: a + "->" + str(b),paths_list[index1]))
                        # Creating a signal_information object for each path
                        # and propagating signal information through the path
                        signal_info = Signal_Information(signal_power, paths_list[index1])
                        self.propagate(signal_info)
                        if signal_info.noise_power != 0:
                            snr = 10 * log(signal_power / signal_info.noise_power)  # Convert to dB
                            accumulated_latency.append(signal_info.latency)    # Appending updated latency for each path
                            accumulated_noise.append(signal_info.noise_power)  # Appending updated noise power for each path
                            accumulated_snr.append(snr)                        # Appending updated snr for each path

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
        print(self.weighted_paths.to_string())

def main():

    signal_power = 1 * 10 ** (-3)
    # Initialization of the Network
    net = Network(file_input)
    net.connect()
    net.draw()
    net.dataframe(signal_power)

if __name__ == '__main__':
    main()
