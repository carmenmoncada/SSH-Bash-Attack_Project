# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import sqrt
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
import sys
from functools import reduce
from scipy.special import erfcinv
from .parameters import c,M
from .math_utils import *
import random

# Allows to show all the lines of the Dataframes
sys.setrecursionlimit(1500)

# CHANGE THE TYPE OF TRANSCEIVER HERE:
transceiver_type = 'Shannon-Rate' # 'Shannon-Rate', 'Flex-Rate', 'Fixed-Rate'

folder_M = f'M={M}'
# Link where the data will be save
root = Path(__file__).parent.parent
# main_folder = root/'Lab3'
# FOR CONGESTION SCENARIO --------------------------------------------------
input_folder = root /'resources'
csv_route_space = root /'Presentation_graphs'      #'Results'
csv_switching_matrix = csv_route_space / transceiver_type / folder_M
file_name = 'full_network.json'            # 'full_network.py', 'not_full_network.py', 'network.json'
file_input = input_folder / file_name

# For Fixed-M scenario------------------------------------------------------
# input_folder = root /'resources'
# csv_route_space = root /'Presentation_graphs'/ 'Fixed M'
# csv_switching_matrix = csv_route_space / transceiver_type
# file_name = 'network.json'    # 'full_network.py', 'not_full_network.py', 'network.json'
# file_input = input_folder / file_name

NUMBER_OF_CHANNELS = compute_number_of_channel(file_input)
#-----------------------------------------------------------------------------------------------------------------------
# Settings for the plots
csfont = {'fontname': 'Times New Roman', 'fontsize': '13', 'fontstyle': 'normal'}
csfont_ = {'fontname': 'Times New Roman', 'fontsize': '14', 'fontstyle': 'normal'}

# Constants of Class Line (physical parameters)
gain_db = 16            # Gain of the amplifier [dB]
noise_figure_db = 5.5   # Noise figure [dB]
alpha_db_km = 0.2       # [db/km]    Attenuation
B2 = 2.13e-26           # [ps^2/km]  Group velocity dispersion coefficient
gamma = 1.27e-3         # [Wm^-1]    non-linearity coefficient (Kerr effect)

# For ASE generation
h = 6.62607015e-34      # Plank constant m^2 kg/s
freq = 193.414e12       # C-band center frequency THZ
bn = 12.5e9             # Noise bandwidth [Hz]

# Constants for class Lightpath
rs = 32e9               # Symbol rate [baud/s][Hz] -> [GHz]
df = 50e9               # Frequency spacing between two consecutive signals [GHz]

# For calculating bit rate
ber = 10e-3             # Bit error rate (BER)

class SignalInformation(object):
    # Class signal information define the attributes of the singal information to be streamed by the connection
    def __init__(self, signal_power=None, path=None):  # constructor
        if signal_power:
            self._signal_power = signal_power   # float
        else:
            self._signal_power = 0
        self._noise_power = 0.0                   # float
        self._latency = 0.0                       # float
        self._isnr = 0.0
        if path:
            self._path = path                   # list[string]
        else:
            self._path = []

    # GETTER -----------------------------------------------------------------------------------------------------------
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

    @property
    def isnr(self):
        return self._isnr

    @isnr.setter
    def isnr(self, value):
        self._isnr = value

    def __repr__(self):
        return f'Nodes(signal_power={self.signal_power}, noise_power={self.noise_power},\
                                     latency={self.latency}, path ={self.path}'

    def update_noise(self, noise):
        # Update the noise power by summing the current noise with the new noise generated in the line
        self.noise_power = self.noise_power + noise
        print("{:.4f}".format(self.noise_power), 'Watts')

    def update_latency(self, latency):
        # Update the noise power by summing the current latency with the new latency generated in the line
        self.latency = self.latency + latency
        print("{:.4f}".format(self.latency), 'ms')

    def update_path(self, path):
        # Remove the first element of the list and update path every time you cross a node
        path.remove(path[0])
        self.path = path
        return self.path


class Lightpath(SignalInformation):
    # Lightpath inherits all the methods and attributes of Signal Information Class
    def __init__(self, signal_power, path, channel):    # constructor of Lightpath
        super().__init__(signal_power, path)
        if signal_power:
            self._signal_power = signal_power  # float
        else:
            self._signal_power = 0
        self._noise_power = 0                           # float
        self._latency = 0                               # float
        self._isnr = 0.0
        if path:
            self._path = path                           # list[string]
        else:
            self._path = []
        self._channel = channel                         # index of the channel
        self._symbol_rate = rs                          # Signal symbol rate Rs [GHz]
        self._df = df                                   # WDM channel spacing [GHz]

    # Getter ----------------------------------------------------------------------------------------------------------
    @property
    def channel(self):
        return self._channel

    @property
    def symbol_rate(self):
        return self._symbol_rate

    @property
    def df(self):
        return self._df

    @property
    def isnr(self):
        return self._isnr

    # SETTER -----------------------------------------------------------------------------------------------------------
    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @symbol_rate.setter
    def symbol_rate(self, value):
        self._symbol_rate = value

    @df.setter
    def df(self, value):
        self._df = value

    @isnr.setter
    def isnr(self, value):
        self._isnr = value


class Nodes(object):
    # Class Node: define all the attributes to define de class node in the network
    def __init__(self, label, position, connected_nodes):
        self._label = label                          # string (attributes)
        self._position = position                    # tuple (float,float)
        self._connected_nodes = connected_nodes      # list[string]
        self._successive = {}                        # dictionary[line]
        self._switching_matrix = None
        self._transceiver = ''

    # GETTER -----------------------------------------------------------------------------------------------------------
    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @property
    def transceiver(self):
        return self._transceiver

    # SETTER -----------------------------------------------------------------------------------------------------------
    @label.setter
    def label(self, label):
        self._label = label

    @position.setter
    def position(self, position):
        self._position = position

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        self._connected_nodes = connected_nodes

    @successive.setter
    def successive(self, value):
        self._successive = value

    @switching_matrix.setter
    def switching_matrix(self, value):
        self._switching_matrix = value

    @transceiver.setter
    def transceiver(self, value):
        self._transceiver = value

    def __repr__(self):
        return f' Nodes(label={self.label}, position={self.position},\
           connected_nodes={self.connected_nodes}, successive ={self.successive})'

    # Define a propagate method that update a lightpath object modifying its path, setting as busy (0) the previous
    # channel and the successive channel of the current channel. (adjacent channels = 0)
    # the update the path and call the successive line.
    def propagate_method(self, node, lightpath, previous_node):  # obj of class signal_info and node
        if len(lightpath.path) <= 1:                             # Condition for only the last node of the path
            print('Propagating:', node.label)
            print('Final Node, propagate ends')

        if len(lightpath.path) > 1:                              # for all other nodes propagate their successive lines
            #print('Propagating:', node.label)
            line = ''.join(lightpath.path[:2])

            if isinstance(lightpath, Lightpath) and previous_node is not None:    # It's None only for the 1st node
                node_out = line[1]
                channels = self.switching_matrix[previous_node][node_out]
                # Comment to consider static switching matrices
                channels[lightpath.channel-1] = 0
                # we subtract one from lightpath.channel to get Channel shape, and we set the adjacent channels
                # equals to 0 to avoid Nonlinear Crosstalk (interference)
                if lightpath.channel != NUMBER_OF_CHANNELS-1:   # If it is not the last channel
                    channels[lightpath.channel-1 - 1] = 0
                elif lightpath.channel != 0:                    # If it’s not the first channel(number of channel!)
                    channels[lightpath.channel-1 + 1] = 0       # adjacent channels cannot be used

            # compare if the second letter of the label == second element of the list path:AB  \
            # [B]==B  path={A,B,D}
            # Updating the  path and propagating over the line
            lightpath.update_path(lightpath.path)
            line_current = node.successive[line]
            # "optimized_launch_power()" set the optimal launch power for each line
            lightpath.signal_power = line_current.optimized_launch_power(lightpath)
            line_current.propagate_method(line_current, lightpath)


class Line(object):
    # Class line define all the attributes for the lines of the network, along with the propagate method to propagate
    # the signal along all the nodes, calling the successive nodes at the end.
    def __init__(self, label, length):
        self._label = label        # string
        self._length = length      # float
        self._successive = {}      # dict[node]
        self._state = np.ones(NUMBER_OF_CHANNELS, dtype=int)        # free=1 or occupied=0
        self._n_amplifiers = np.ceil(self._length/80e3) + 1         # one every 80 km + 1 more to have some slack
        self._gain = gain_db                                        # Gain [dB]
        self._noise_figure = noise_figure_db                        # Noise Figure [dB]
        self._alpha = alpha_db_km*1e-3 / (20 * np.log10(np.e))      # Alpha (linear value)
        self._beta2 = B2                                            # |B2| [s^2/km]
        self._gamma = gamma                                         # gamma(Wm)^-1
        self._n_span = self.n_amplifiers - 1

    # GETTER -----------------------------------------------------------------------------------------------------------
    @property
    def label(self):
        """Label of the line"""
        return self._label

    @property
    def length(self):
        """Length of the line"""
        return self._length

    @property
    def successive(self):
        """Successive Node"""
        return self._successive

    @property
    def state(self):
        """Channels of the Line"""
        return self._state

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def gain(self):
        return self._gain

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta2(self):
        return self._beta2

    @property
    def gamma(self):
        return self._gamma

    @property
    def n_span(self):
        return self._n_span

    # SETTER ----------------------------------------------------------------------------------------------------------
    @label.setter
    def label(self, value):
        self._label = value

    @length.setter
    def length(self, value):
        self._length = value

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @state.setter
    def state(self, state):
        self._state = state

    @noise_figure.setter
    def noise_figure(self, value):
        self._noise_figure = value

    @gain.setter
    def gain(self, value):
        self._gain = value

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @beta2.setter
    def beta2(self, value):
        self._beta2 = value

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @n_span.setter
    def n_span(self, value):
        self._n_span = value

    def __repr__(self):
        return f'Line(label={self.label}, length={self.length}, successive ={self.successive})'

    def latency_generation(self, length):
        n = 1                # Glass:1.52 index of refraction
        # c = 3 * (10 ** 8)  # Speed of light 3.10^8
        return (length / (2/3 * c)) * n

    def noise_generation(self, signal_power):
        #total_noise = 10**(-9) * signal_power * length
        # (Pase + Pnli) / Pch
        ase_noise = self.ase_generation()           # amplified spontaneous emission
        nli_noise = self.nli_noise(signal_power)    # non-linear interference
        total_noise = ase_noise + nli_noise
        return total_noise

    # Propagate method, propagate the signal power ALONG THE LINES, UPDATING THE SIGNAL_INFORMATION and the ISNR on
    # the lightpath, after that it calls the successive node to propagate the signal over the node.
    def propagate_method(self, line, signal_information):      # Propagates the signal information along the line
        # print('Propagating Line', line.label)                # it implies the signal of the lightpaths too
        # Generating latency and noise
        latency = line.latency_generation(line.length)
        # noise = line.noise_generation(signal_information.signal_power, line.length)  # for old noise_generation
        noise = line.noise_generation(signal_information.signal_power)                 # pass the signal power directly

        # updating values for latencies and noise
        signal_information.update_latency(latency)
        signal_information.update_noise(noise)

        # Updating ISNR for the lightpath adding the isnr of the current line
        self.update_isnr(signal_information)

        previous_node = line.label[0]
        for node in line.successive:
            if node == line.label[1]:  # line.successive is a instance Node 'B' == [B]
                line.successive[node].propagate_method(line.successive[node], signal_information, previous_node)
            break

    # Evaluate the total amount of Amplified Spontaneous Emissions (ASE) for the cascade of amplifiers
    def ase_generation(self):
        N_ampl = self.n_amplifiers                                # Number of amplifiers
        noise_figure_linear = db_to_linear(self.noise_figure)     # Noise figure nf
        gain_linear = db_to_linear(self.gain)                     # Gain
        ase_noise = N_ampl * (h * freq * bn * noise_figure_linear * abs(gain_linear - 1))
        return ase_noise

    '''Observation:
           NLI increases with:
           - Reduction of chromatic dispersion (beta2)
           - Reduction of fiber loss (alfa)
           - Reduction of channel spacing (df)
           - Increase of nonlinear coefficient (gamma)
           - Increase of number of channel, even if weak (n_channel)'''
    # Evaluate the total amount generated by the nonlinear interface noise
    def nli_noise(self, signal_power):
        nli_noise = signal_power**3 * self.n_span * self.eta_nli_generation() * bn
        # Non-linear interference power NLI = Pch^3 * eta_nli * n_span * bn
        return nli_noise

    # Evaluate the total amount of NLI generated by the non-linear interface noise
    def eta_nli_generation(self):
        leff = 1 / (2 * self.alpha)         # extreme case when leff <= 1/2alpha
        eta_nli = (16/(27*np.pi)) * np.log(
            (np.pi**2/2) * ((abs(self.beta2) * rs**2)/self.alpha) * NUMBER_OF_CHANNELS**(2*rs/df))\
                  *(self.alpha/abs(self.beta2)) * ((self.gamma**2 * leff**2)/rs**3)
        return eta_nli

    # Find out the optimized power
    def optimized_launch_power(self, lightpath):
        ase = self.ase_generation()
        eta_nli = self.nli_noise(lightpath.signal_power)
        optimal_power = (ase/(2*eta_nli*self.n_span*bn))**(1/3)
        return optimal_power

    def gsnr_generation(self, lightpath):
        gsnr = lightpath.signal_power/self.noise_generation(lightpath.signal_power)
        return gsnr

    # Generate the ISNR by computing the inverse of GSNR
    def isnr_generation(self, signal_information):
        isnr = 1 / self.gsnr_generation(signal_information)
        return isnr

    # update the ISNR of the signal information
    def update_isnr(self, signal_information):
        signal_information.isnr += self.isnr_generation(signal_information)
        return signal_information.isnr


# Network class contain all the methods needed to construct the network, defining the route space and switching matrix
# of each node. Class Network composed of nodes (ROADMs) and lines (fiber lines)
class Network(object):
    def __init__(self, file):  # received instance of calls line and node
        self.nodes = {}        # dictionary for nodes
        self.lines = {}        # dictionary for lines
        self.weighted_paths = pd.DataFrame(columns=['Paths', 'Accumulated Latency [s]', 'Accumulated Noise [W]',
                                                    'Accumulated SNR [dB]'])

        with open(file, 'r') as f:
            data = json.load(f)
            # creating a node instance for each node in the file
            self.nodes = {i: Nodes(str(i), data[i]['position'], data[i]['connected_nodes']) for i in data}

            # Assigning for each key ('A','B','C'...) a Node instance with
            # all its attributes (label, position, connected lines)
            for node in self.nodes:         # Assigning to each line their successive nodes
                for j in range(len(self.nodes[node].connected_nodes)):
                    # Obtaining the position of Node connected with A, in the first  case B,C,D (Euclidean distance)
                    x1 = self.nodes[node].position[0]
                    x2 = self.nodes[self.nodes[node].connected_nodes[j]].position[0]
                    y1 = self.nodes[node].position[1]
                    y2 = self.nodes[self.nodes[node].connected_nodes[j]].position[1]
                    length = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
                    # Creating the lines, sending the lables of the lines and length of the distances between the nodes
                    node_connected = self.nodes[node].connected_nodes[j]
                    self.lines[node + self.nodes[node].connected_nodes[j]] = Line(node + node_connected, length)

            # Check whether the json file contains the 'Transceiver Information' and 'Switching Matrix' info
            for actual_node in self.nodes:
                if 'transceiver' in data[actual_node].keys():
                    self.nodes[actual_node].transceiver = data[actual_node]['transceiver']
                else:
                    self.nodes[actual_node].transceiver = transceiver_type

                if 'switching_matrix' in data[actual_node].keys():
                    # Initializing switching matrix
                    self.nodes[actual_node].switching_matrix = data[actual_node]['switching_matrix']

                    columns = [str(i) for i in range(1, NUMBER_OF_CHANNELS + 1)]
                    # Concatenating the Paths columns with the number of channels column list
                    self.route_space = pd.DataFrame(columns=['Paths'] + columns)

                else:
                    connected_node_list =  data[actual_node]['connected_nodes']
                    self.nodes[actual_node].switching_matrix = self.switching_matrix_initial(connected_node_list)

                    columns = [str(i) for i in range(1, NUMBER_OF_CHANNELS + 1)]
                    # Concatenating the Paths columns with the number of channels column list
                    self.route_space = pd.DataFrame(columns=['Paths'] + columns)


    # SECOND: Setter and Getter-----------------------------------------------------------------------------------------
    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_path(self):
        return self._weighted_path

    @property
    def route_space(self):
        return self._route_space

    # SETTER -----------------------------------------------------------------------------------------------------------
    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @lines.setter
    def lines(self, value):
        self._lines = value

    @weighted_path.setter
    def weighted_path(self, value):
        self._weighted_path = value

    @route_space.setter
    def route_space(self, value):
        self._route_space = value

    def __repr__(self):
        return f'{self.lines},\n {self.nodes})'

    # Connect method set the successive attributes of all Nodes as dictionaries
    # each node must contain a dictionary of lines and vice-versa
    def connect(self,):
        # Method to assign the successive element for each instance Node and Line
        for node in self.nodes:
            for line in self.lines:
                if line.startswith(node):
                    for i in range(len(line)):  # to avoid adding root node in the successive node dictionary
                        if i != 0:
                            # Assigning to the Nodes and Lines the successive elements (nodes and lines)
                            self.nodes[node].successive[line] = self.lines[line]        # 'A'.successive = 'AB'
                            self.lines[line].successive[line[i]] = self.nodes[line[i]]  # 'AB'.successive = 'B'

    # Initializing the switching matrix
    # Global variables file_input and file_name
    def switching_matrix_initial(self, lista_connected):
        # old files 'nodes.json', 'nodes_full.json', 'nodes_not_full.json'
        # For Switching Matrix initialization when the file doesn't provide any SW matrix
        sw2_dict = {}
        # Creating the keay C:{C=[],B=[],F=[],E=[]}
        for key_node in lista_connected:
            sw1_dict = {}
            for k in lista_connected:
                if key_node == k:
                    sw1_dict[self.nodes[k].label] = np.zeros(NUMBER_OF_CHANNELS, dtype=int)
                elif key_node != k:
                    # Initializing the switching_matrix when the pairs of nodes are different
                    sw1_dict[self.nodes[k].label] = np.ones(NUMBER_OF_CHANNELS, dtype=int)

            sw2_dict[key_node] = sw1_dict  # {'C':{C:[0000], B=[1111],...}}
        return sw2_dict             # it returns a dictionary


    # Find_paths: given two node labels, returns all paths that connect the 2 nodes
    # as a list of node labels. Admissible path only if cross any node at most once.
    def find_path(self, node_input, node_output, lista):
        list_paths = lista + [node_input]     # list of all possible paths for label1 [A]
        if node_input == node_output:
            return [list_paths]
        if node_input not in self.nodes:      # Check if the label is defined in nodes
            return []
        paths = []                            # List of list with paths
        for node in self.nodes[node_input].connected_nodes:
            if node not in list_paths:
                # If node is not in the list, it appends the node in the list and search the next one recursively
                new_path = self.find_path(node, node_output, list_paths)
                for path in new_path:
                    paths.append(path)
        return paths

    # Propagate lightpath through path specified in it,
    # and returns the modified spectral information.
    def propagate(self, lightpath):
        for node in self.nodes:
            node_path = lightpath.path
            if node == node_path[0]:                 # if node is equal to the first element of the list path
                print('-------------', 'Propagation Starts', '-------------')
                self.nodes[node].propagate_method(self.nodes[node], lightpath, previous_node=None)
                break
        return lightpath

    # Propagate signal_information through path specified in it
    # and it returns the modified spectral information
    def propagate_probe(self, signal_information):
        for node in self.nodes:
            node_path = signal_information.path
            if node == node_path[0]:
                print('-------------', 'Propagation Starts', '--------------')
                self.nodes[node].propagate_method(self.nodes[node], signal_information, None)
                break
        return signal_information

    # Draw the Network evaluated, taking into account the position of each node
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
                    plt.xlabel('X', **csfont)
                    plt.ylabel('Y', **csfont)
                    plt.title('Network Topology', **csfont)
                    plt.tight_layout()
        plt.style.use('seaborn-paper')
        plt.show()

    # Method to create the weighted paths dataframe, where the SNR, latency and bit rate are shown for each possible
    # path within the network, appending the accumulated values of each metric.
    # It also initializes the Route Space by setting all the channels of each path in '1'.
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
                        # Using Signal Information of 1mW
                        signal_info = SignalInformation(signal_power, paths_list[index1])
                        signal_info = self.propagate_probe(signal_info)      # probe method is a propagate method
                        if signal_info.noise_power != 0:                     # but using the signal information
                            snr = 10 * np.log10(signal_info.signal_power / signal_info.noise_power)  # Convert to dB
                            accumulated_latency.append(signal_info.latency)   # Appending updated latency for each path
                            accumulated_noise.append(signal_info.noise_power)  # Appending updated noise power e/path
                            accumulated_snr.append(snr)                       # Appending updated snr for each path
                        else:
                            print('Error in the noise power, line 598')
        # Adding lists of paths, accu_latency, accu_noise and SNR values for each path in the df
        self.weighted_paths['Paths'] = path_df
        self.weighted_paths['Accumulated Latency [s]'] = accumulated_latency
        # Apply scientific notation format to accumulated_noise
        self.weighted_paths['Accumulated Noise [W]'] = pd.Series(accumulated_noise).apply(lambda x: '{:e}'.format(x))
        self.weighted_paths['Accumulated SNR [dB]'] = accumulated_snr
        # For Route_Space DF
        self.route_space['Paths'] = path_df
        # Set all values in columns '1' to '10' to 1 (Initializing DF)
        columns = [str(i) for i in range(1, NUMBER_OF_CHANNELS+1)]
        self.route_space[columns] = 1    # free = 1 or occupied=0

        # Permanently changes the pandas settings
        pd.set_option('display.max_rows', None)      # Show all the rows of the dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Save the DataFrame to a CSV file
        self.weighted_paths.to_csv(csv_switching_matrix/'switching_matrix.csv', index=False)
        print(self.weighted_paths.to_string())

    def find_best_snr(self, node_input, node_output):
        # Returns a path with best SNR found and update the channels occupancy of each line of the path found.
        # Getting all the possible paths between node1 and node2
        paths_btw_nodes_df = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node_input) &
                                                 self.weighted_paths['Paths'].str.endswith(node_output)]

        max_snr = (-5000)
        best_path = 'NaN'
        best_path_index = -1
        index_channel = -1

        for index, row in paths_btw_nodes_df.iterrows():
            for channel in range(1, NUMBER_OF_CHANNELS+1):

                current_snr = row['Accumulated SNR [dB]']
                idx = [idx for idx, col in enumerate(self.route_space.columns) if str(channel) == col]
                column_channel = str(idx[0])
                # Compare if the current SNR is the greater than the Max SNR set before, if so, save the path,
                # update the max_snr and the index_channel
                if (current_snr > max_snr) and (self.route_space.loc[index, column_channel] == 1):
                    max_snr = current_snr
                    best_path = row['Paths']
                    best_path_index = index
                    index_channel = channel

        if index_channel is not -1:
            lines_current_list = list()
            resulting_path = best_path.split('->')
            # "split" uses -> like a flag to create different strings
            # example: A->B->C->D become ABCD
            # creating the list of paths
            for i in range(len(resulting_path)):
                last_elem = resulting_path[len(resulting_path) - 1]
                first_elem = resulting_path[i]
                if first_elem != last_elem:
                    # Creating the label of the line
                    line = resulting_path[i] + resulting_path[i + 1]
                    lines_current_list.append(line)
            # Iterates over all the lines within the path
            # Changing the current state of channels throughout the lines of the path
            for line in lines_current_list:
                self.lines[line].state[index_channel-1] = 0       # free=1 or occupied=0

        return best_path_index, index_channel

    def find_best_latency(self, node_input, node_output):
        # Returns the path with the best latency (min latency)
        # Filtering the df Path column, to find the required path
        filter_path = self.weighted_paths[self.weighted_paths['Paths'].str.startswith(node_input) &
                                          self.weighted_paths['Paths'].str.endswith(node_output)]

        best_latency = float('inf')
        best_path = 'NaN'
        best_path_index = -1
        index_channel = -1

        for index, row in filter_path.iterrows():
            for channel in range(1, NUMBER_OF_CHANNELS+1):

                current_latency = row['Accumulated Latency [s]']
                idx = [idx for idx, col in enumerate(self.route_space.columns) if str(channel) == col]
                column_channel = str(idx[0])
                if (current_latency < best_latency) and (self.route_space.loc[index, column_channel] == 1):
                    best_latency = current_latency
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
                self.lines[line].state[index_channel-1] = 0           # free=1 or occupied=0

        return best_path_index, index_channel

    def route_space_update(self,):
        # Updates all the paths in the route_space performing for each path the multiplication of all the line
        # states and all switching matrix array that compose the path (excluding the switching matrix of the
        # initial and last nodes).
        paths = self.weighted_paths['Paths']
        index = 0
        for path in paths:
            channel = np.ones(NUMBER_OF_CHANNELS, dtype=int)
            path_str = str(path).split('->')
            for i in range(len(path_str)):
                # Access to the lines that does not contain First node and Last node
                # ABCD --> BC = [1,1,1,1,1,1] * swithcing_matrix [0,1,1,1,1,1]
                if i != 0 and (i != len(path_str)-1) and i % 2 == 0:
                    channel *= self.lines[path_str[i-1] + path_str[i]].state
                    channel *= self.nodes[path_str[i]].switching_matrix[path_str[i-1]][path_str[i+1]]
                # For the lines in the middle of the path and does not contain the Last Node
                elif (i != len(path_str)-1) and i % 2 != 0:
                    channel *= self.lines[path_str[i-1] + path_str[i]].state    # Access to the state of the line
                    channel *= self.nodes[path_str[i]].switching_matrix[path_str[i-1]][path_str[i+1]]
                elif i != 0:
                    # Access to the last line channels and remain unchanged
                    channel *= self.lines[path_str[i-1] + path_str[i]].state

            # Updating channel availability of Route Space DF
            self.route_space.iloc[index, 1:] = channel
            index += 1

    # Method that given a list of connection, streams them and creates
    # the connections assigning to them the best available path.
    # Label: is the preference between SNR and Latency
    def stream(self, connection_list, signal_power, label):
        best_path_index = ''
        index_channel = ''

        for connection in range(len(connection_list)):
            if (label == 'latency') or (label == 'Latency'):
                best_path_index, index_channel = self.find_best_latency(connection_list[connection].input,
                                                                        connection_list[connection].output)
            elif (label == 'snr') or (label == 'SNR'):
                # For the SNR
                best_path_index, index_channel = self.find_best_snr(connection_list[connection].input,
                                                                    connection_list[connection].output)
            else:
                print(f'\n Error in the type of connection requested, select "Latency" or "SNR"')

            if best_path_index != -1 and index_channel != -1:
                best_path = self.weighted_paths.loc[best_path_index, 'Paths']
                resulting_path = str(best_path).split('->')      # ['A'->'C'->'E']->['A','C','E']

                # Propagating throughout the lightpath the signal information, using the resulting path
                # from the best SNR or latency
                lightpath = Lightpath(signal_power, resulting_path, index_channel)
                # Calculating the bit rate, using the transceiver type of the first node
                first_node = self.nodes[best_path[0]]
                # Calculate the bit rate based on the transceiver of the FIRST NODE
                connection_bit_rate = self.calculate_bit_rate(lightpath, first_node.transceiver)

                if connection_bit_rate != 0:             # If the connection is not rejected we propagate
                    self.propagate(lightpath)
                    connection_list[connection].latency = lightpath.latency
                    connection_list[connection].snr = 10 * np.log10(1/lightpath.isnr)  # GSNR = 1/ISNR [dB]
                    connection_list[connection].bit_rate = connection_bit_rate
                    # Updating the rout_space dataframe by searching using the index path and the index_channel=column
                    self.route_space_update()
                else:
                    # For connections rejected
                    connection_list[connection].latency = None
                    connection_list[connection].snr = None  # In dB
                    print('Connection rejected because of bit rate is 0')
            else:
                # When there is no path available case
                connection_list[connection].latency = 0
                connection_list[connection].snr = None    # In dB
                print('Connection rejected because there is no path available')

        # Save the DataFrame to a CSV file
        self.route_space.to_csv(csv_switching_matrix/'route_space.csv', index=False)
        print(self.route_space.to_string())

    def calculate_bit_rate(self, lightpath, strategy):
        # Method used to calculate the maximum obtainable bit rate for a lightpath given the transceiver type.
        bit_rate = 0                # bit rate Rb
        # rs = 32e9           # symbol rate Rs
        # bn = 12.5e9         # bandwidth noise Bn
        # ber = 10e-3         # bit error rate BERt

        path = "".join(lightpath.path)  # Lightpath "path" is a list, with join I can obtain a string with no space
        path_with_arrows = ""
        for i in path:
            if i != path[-1]:
                path_with_arrows += i + "->"
            else:
                path_with_arrows += i

        gsnr_db = float(self.weighted_paths.loc[self.weighted_paths['Paths'] == path_with_arrows,
                                          'Accumulated SNR [dB]'].values)         # Is in decibel
        gsnr = db_to_linear(gsnr_db)      # 10**(gsnr_db/10) Converting to linear
        rs = lightpath.symbol_rate

        if strategy == 'Flex-Rate' or strategy=='flex-rate':       # PM-QPSK modulation
            if gsnr < 2 * np.square(erfcinv(2 * ber)) * rs/bn:
                bit_rate = 0
            # PM-QPSK modulation
            if 2 * np.square(erfcinv(2 * ber)) * rs/bn <= gsnr < 14/3 * np.square(erfcinv(3/2 * ber)) * rs/bn:
                bit_rate = 100e9
            # PM-8QAM modulation
            if 14/3 * np.square(erfcinv(3/2 * ber)) * rs/bn <= gsnr < 10 * np.square(erfcinv(8/3 * ber)) * rs/bn:
                bit_rate = 200e9
            # PM-16QAM modulation
            if gsnr >= 10 * np.square(erfcinv(8/3 * ber)) * rs/bn:
                bit_rate = 400e9
        elif strategy == 'shannon-rate' or strategy == 'Shannon-Rate':
            bit_rate = 2 * rs * np.log2(1 + gsnr * (rs/bn))
        else:
            # Fixed-rate default
            if gsnr >= 2 * np.square(erfcinv(2 * ber)) * rs/bn:
                bit_rate = 100e9
            else:
                bit_rate = 0
        return bit_rate

    def request_generation_traffic_matrix(self, traffic_matrix, connections_list, signal_power, label,
                                          count_request, rejected_connections):
        # Method that manages the traffic requested with a traffic matrix
        nodes = list(self.nodes.keys())
        # Creating a random connection using two nodes from the nodes list
        node_in = random.choice(nodes)
        node_out = random.choice(tuple(self.nodes.keys() - {node_in}))

        # Condition to Stop : Generating connections only for input_node != output_node and
        # if the element corresponding to input, output node in the traffic matrix
        # has still available traffic to allocate, otherwise Saturation is set in -1
        if traffic_matrix[node_in][node_out] != 0 and node_in != node_out and traffic_matrix[node_in][node_out] != -1:

            connection = Connection(node_in, node_out, signal_power)
            self.stream([connection], signal_power, label)
            connections_list.append(connection)

            # Updating traffic matrix if the best path is found in the stream() method
            if connection.snr != 0 and connection.snr is not None:
                # Case 1: When the bit rate requested by the connection is completely satisfied
                if connection.bit_rate >= traffic_matrix[node_in][node_out]:
                    # Saturation of the matrix use flag -1
                    traffic_matrix[node_in][node_out] = -1
                    count_con = count_request + 1
                    # decrement, because the capacity was guaranteed for that connection
                    return 1, count_con, rejected_connections
                else:
                    # Case 2 :Updating the remaining capability after having satisfied the request for
                    # the current connection
                    # allocated traffic (connection.bit_rate) has to be subtracted to the given traffic matrix.
                    traffic_matrix[node_in][node_out] -= connection.bit_rate
                    count_con = count_request + 1
                    return 0, count_con, rejected_connections  # Don't decrement
            else:
                # Case 3 : If there is no found any suitable path for that connection
                #traffic_matrix[node_in][node_out] = -1
                # If is not possible to define a suitable path for the connection,
                # decrement the number of all possible connections by returning 1
                print('No traffic allocation because SNR of [{}->{}] is 0'.format(node_in, node_out))
                rejected_connections += 1
                return 1, count_request, rejected_connections    # decrement
        else:
        # When the matrix is already saturated
            rejected_connections += 1
            return 0, count_request, rejected_connections  # Default return


class Connection(object):

    def __init__(self, input_node, output_node, signal_power):
        self._input = input_node             # string
        self._output = output_node           # string
        self._signal_power = signal_power    # float
        self._latency = 0.0                  # float
        self._snr = 0.0                      # float
        self._bit_rate = 0                   # int

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate

    def __repr__(self):
        return f'{self.input}, {self.output}, {self.signal_power}, {self.latency}, {self.snr}, {self.bit_rate}'
