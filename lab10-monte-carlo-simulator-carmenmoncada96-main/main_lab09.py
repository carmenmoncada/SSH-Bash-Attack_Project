# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from core.elements import *
from core.parameters import *
from core.utils import *
import random


def main():
    # Check the parameters.py file to check the constants defined
    # such as speed of light, signal power, number of connections, etc.

    # Initialization of the Network
    net = Network(file_input)   # Check in elements.py to see the root of the file
    net.connect()
    net.switching_matrix_initial(file_input, file_name)
    # net.draw()
    net.weighted_paths_dataframe(signal_power)

    # Set a seed for the random number generator
    random.seed(1)

    # Stream call
    # I recall the method stream now for the SNR (100 instances)-------------------------------------------------------
    label = 'snr'           # choose the label for stream method between 'snr' or 'latency'
    option = 'fixed_list'   # choose between random or fixed_list
    lista_connection = []
    connection_list = []

    if option == 'random':
        # Creating 100 connections with signal_power equal to 1 and with input/output nodes
        # randomly chosen.
        for i in range(connections):
            node1 = random.choice(list(net.nodes.keys()))
            node2 = random.choice(tuple(net.nodes.keys() - {node1}))
            if node1 != node2:
                # For stream
                connection_list.append(Connection(node1, node2, signal_power))
                # For the dataframe
                nro_connection = [Connection(node1, node2, signal_power)]
                lista_connection.append(nro_connection)

        # Generate the CSV for the connection list
        generate_connection_csv(lista_connection)
        # Streaming the signal
        net.stream(connection_list, signal_power, label)

    elif option == 'fixed_list':
        # Creating 100 connections with signal_power equal to 1 and with input/output nodes
        # randomly chosen beforehand and saved in 'lista_connection.csv'
        file_path = csv_switching_matrix/'lista_connection.csv'
        connection_df = pd.read_csv(file_path, delimiter=',', header='infer', names=None, skiprows=None)
        for index, row in connection_df.iterrows():
            connection_list.append(Connection(row['input'], row['output'], row['signal_power']))
        # Streaming the signal
        net.stream(connection_list, signal_power, label)

    # Calling best_metric function to get the best snr, latency and bit rate
    best_snr, best_latency, best_bit_rate = best_metrics(connection_list, net)

    # Converting into arrays the snr, latency and bit rates
    snr_array = [-5 if value is None else value for value in best_snr]
    snr_array = np.array(snr_array)
    latency_array = np.array(best_latency)
    bit_rate_array = np.array(best_bit_rate)

    # Computing the capacity and average bit rate
    total_capacity, average_bit_rate = total_avg_max_min_capacity(bit_rate_array)
    plots(label, latency_array, snr_array, bit_rate_array, average_bit_rate, total_capacity)


if __name__ == '__main__':
    main()
