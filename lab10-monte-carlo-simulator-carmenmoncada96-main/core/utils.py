# Use this file to define your generic methods, e.g. for plots
import math
from .elements import plt
from .elements import csfont, csfont_
import seaborn as sns
from .math_utils import *
from .elements import csv_route_space, csv_switching_matrix
from .elements import pd, Connection
from .parameters import BIT_RATE_REQUEST, connections_number
import random
import os


def truncate(number, decimals=0):
    """
    Truncate or round a number to a specified number of decimal places.
    Parameters:
    - number: The number to be truncated or rounded.
    - decimals: The number of decimal places to keep (default is 0)."""
    if not isinstance(decimals, int):
        raise TypeError('The decimal number must be a integer.')
    elif decimals < 0:
        raise TypeError('The number of decimal must be a number greater than 0')
    elif decimals == 0:
        return math.trunc(number)

    # Returns:
    # The truncated or rounded number. 
    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


def total_avg_max_min_capacity(bit_rate_array, snr_array):
    total_cap = np.sum(bit_rate_array)
    total_capacity = truncate(total_cap / 1e9, 3)

    bit_rate_avg = np.mean(bit_rate_array)
    bit_rate_average = truncate(bit_rate_avg / 1e9, 3)

    max_cap = np.max(bit_rate_array)    # Max bit rate
    max_capacity = truncate(max_cap / 1e9, 3)

    min_cap = np.min(bit_rate_array)    # Min bit rate
    min_capacity = truncate(min_cap / 1e9, 3)

    max_snr = np.max(snr_array)
    min_snr = np.min(snr_array[snr_array>0])

    return total_capacity, bit_rate_average, max_capacity, min_capacity, max_snr, min_snr


def print_best_metrics(lista_connection, net):
    # Selecting best snr--------------------------------------------------------------------------------------
    best_snr = []
    path_snr = []
    print('\n''\n''Best SNR found')
    for i in range(len(lista_connection)):
        if lista_connection[i].snr is None:
            print(f'For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
                  lista_connection[i].snr, 'Connection rejected because there is not path available')
        else:
            print(f'For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
                  lista_connection[i].snr)
        best_snr.append(lista_connection[i].snr)     # IN dB
        path_snr.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Selecting the best latency------------------------------------------------------------------------------
    best_latency = []
    path_latency = []
    print('\n''\n''Best latency found')
    for i in range(len(lista_connection)):
        if lista_connection[i].latency is None:
            print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
                  lista_connection[i].latency, 'Connection rejected because the bit rate is 0 Gbps')

        elif lista_connection[i].latency == 0:
            print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
                  lista_connection[i].latency, 'Connection rejected because there is not path available')
        else:
            print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
                  lista_connection[i].latency)

        best_latency.append(lista_connection[i].latency)
        path_latency.append(lista_connection[i].input + '->' + lista_connection[i].output)

    # Selecting Bit Rates-------------------------------------------------------------------------------------
    best_bit_rate = []
    path_bit_rate = []
    print('\n''\n''Best bit rate found')
    for i in range(len(lista_connection)):
        first_node = net.nodes[lista_connection[i].input]
        print('For a path between: [', lista_connection[i].input, '->', lista_connection[i].output, '] is',
              lista_connection[i].bit_rate, 'with the strategy: ', '"', first_node.transceiver, '"')
        best_bit_rate.append(lista_connection[i].bit_rate)
        path_bit_rate.append(lista_connection[i].input + '->' + lista_connection[i].output)

    return best_snr, best_latency, best_bit_rate, path_bit_rate


def plots(latency_array, snr_array, bit_rate_array, average_bit_rate, total_capacity, max_capacity, min_capacity,
          transceiver_type, rejected_con, avg_latency, avg_snr, M, count_request):

    # Plotting SNR distributions ------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(snr_array, color='orange', kde=False, bins=30, edgecolor='k', fill=True)

    snr_scaled = "{:.4g}".format(avg_snr)
    # Create custom legend items with additional information
    legend_items = [f'Average SNR: {snr_scaled} dB',
                    f'Negative SNRs represent the connections rejected']
    # Add the legend as a text box in the plot
    legend_text = '\n'.join(legend_items)
    plt.text(0.02, 0.92, legend_text, transform=plt.gca().transAxes, fontsize=10, va='center', ha='left',
             fontname='Times New Roman', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square, pad=0.3'))
    plt.xlabel('SNR [dB]', **csfont)
    plt.ylabel('Occurrences', **csfont)

    # Case of lab 9
    # plt.title(f'SNR for {connections_number} Connections for "{transceiver_type}" case', **csfont)
    plt.title(f'SNR with "{transceiver_type}" Strategy', **csfont)
    plt.tight_layout()
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')

    # Ensure base_path directory exists
    base_path = csv_switching_matrix
    os.makedirs(base_path, exist_ok=True)
    # Construct the full path for saving the plot
    log_file = fr'snr_{M}.png'
    full_path_1 = os.path.join(base_path, log_file)
    # Save the plot as a PNG file
    try:
        plt.savefig(full_path_1)
        # print(f"Plot saved successfully at: {full_path_1}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        # Clear the current figure to avoid overlap in future plots
        plt.clf()
        plt.close()

    # Latency distributions ------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(latency_array, color='red', kde=False, bins=30, edgecolor='k', fill=True)
    # Create custom legend
    value_in_ms = avg_latency * 1000    # to convert it into ms
    # Format the average latency value
    latency_ms = "{:.5g}".format(value_in_ms)
    legend_items = [f'Average latency: {latency_ms} ms']
    # Add the legend as a text box in the plot
    legend_text = '\n'.join(legend_items)
    plt.text(0.02, 0.92, legend_text, transform=plt.gca().transAxes, fontsize=10, va='center', ha='left',
             fontname='Times New Roman', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square, pad=0.3'))

    # Case of lab 9
    # plt.title(f'Latencies for {connections_number} Connections for "{transceiver_type}" case', **csfont)
    plt.title(f'Latencies with "{transceiver_type}" Strategy', **csfont)
    plt.xlabel('Latency [s]', **csfont)
    plt.ylabel('Occurrences', **csfont)
    plt.tight_layout()
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')

    # Ensure base_path directory exists
    base_path = csv_switching_matrix
    os.makedirs(base_path, exist_ok=True)
    # Construct the full path for saving the plot
    log_file = fr'latency_{M}.png'
    full_path_1 = os.path.join(base_path, log_file)
    # Save the plot as a PNG file
    try:
        plt.savefig(full_path_1)
        # print(f"Plot saved successfully at: {full_path_1}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        # Clear the current figure to avoid overlap in future plots
        plt.clf()
        plt.close()

    # Plotting BIT RATE distributions ------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    # Bit Rates
    sns.histplot(bit_rate_array, color='blue', kde=False, bins=30, edgecolor='k', fill=True)
    plt.xlabel('Bit Rate', **csfont)
    plt.ylabel('Occurrences', **csfont)
    # Case of lab 9
    # plt.title(f'Bit Rate for {connections_number} Connections with "{transceiver_type}" case ', **csfont)
    plt.title(f'Bit Rates with "{transceiver_type}" Strategy ', **csfont)

    # Create custom legend items with additional information
    legend_items = [f'Average Bit Rate: {average_bit_rate} Gbps',
                    f'Total capacity allocated: {total_capacity} Gbps',
                    f'Maximum capacity: {max_capacity} Gbps',
                    f'Minimum capacity: {min_capacity} Gbps',
                    f'Number of rejected connections: {rejected_con}',
                    f'Number of connection accepted: {count_request}']
                    # fr'Number of blocking events :{blocking_events}']

    # Add the legend as a text box in the plot
    legend_text = '\n'.join(legend_items)
    plt.text(0.02, 0.87, legend_text, transform=plt.gca().transAxes, fontsize=10, va='center', ha='left',
             fontname='Times New Roman', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.3'))
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')
    plt.tight_layout()
    plt.style.use('seaborn-paper')

    # Ensure base_path directory exists
    base_path = csv_switching_matrix
    os.makedirs(base_path, exist_ok=True)
    # Construct the full path for saving the plot
    log_file = fr'bit_rate_{M}.png'
    full_path_3 = os.path.join(base_path, log_file)
    # Save the plot as a PNG file
    try:
        plt.savefig(full_path_3)
        # print(f"Plot saved successfully at: {full_path_3}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        # Clear the current figure to avoid overlap in future plots
        plt.clf()
        plt.close()
    # plt.show()


def generate_connection_csv(lista_connection): #transceiver_type, M):
    attribute_names = ['input', 'output', 'signal_power', 'latency', 'snr', 'bit_rate']
    # Create a list of lists using list comprehension
    list_of_lists = [[getattr(obj[0], attr) for attr in attribute_names] for obj in lista_connection]
    connection_df = pd.DataFrame(list_of_lists, columns=attribute_names)

    # Save the plot with a dynamic filename
    filename = csv_switching_matrix/'lista_connection.csv'
    # Save the DataFrame to a CSV file
    connection_df.to_csv(filename, index=False)


def uniform_traffic_matrix_generation(network, M):  # remove M entry if you are using main.py
    uniform_traffic_matrix = {}     # dict
    for node_in in network.nodes.keys():
        uniform_traffic_matrix[node_in] = {}
        for node_out in network.nodes.keys():
            if node_in != node_out:
                uniform_traffic_matrix[node_in][node_out] = BIT_RATE_REQUEST * M
            else:
                uniform_traffic_matrix[node_in][node_out] = 0

    return uniform_traffic_matrix


def generating_connections(option, connection_list, net, signal_power, label):
    # GENERATING 100 RANDOM CONNECTIONS --------------------------------------------------------------------------------
    # I recall the method stream now for the SNR (100 instances)
    if option == 'random':
        # Creating 100 connections with signal_power equal to 1 and with input/output nodes
        # randomly chosen.
        lista_connection = []  # list to append all the connections in a csv
        for i in range(connections_number):
            node1 = random.choice(list(net.nodes.keys()))
            node2 = random.choice(tuple(net.nodes.keys() - {node1}))
            if node1 != node2:
                # For stream
                connection_list.append(Connection(node1, node2, signal_power))
                # For the DataFrame and CSV
                nro_connection = [Connection(node1, node2, signal_power)]
                lista_connection.append(nro_connection)

        # Generate the CSV for the connection list
        generate_connection_csv(lista_connection)
        # Streaming the signal
        # recalling the method stream now for the SNR (100 instances)
        net.stream(connection_list, signal_power, label)

    elif option == 'fixed_list':
        # Creating 100 connections with signal_power equal to 1mW and with input/output nodes
        # randomly chosen beforehand and saved in 'lista_connection.csv'
        file_path = csv_switching_matrix/'lista_connection.csv'
        connection_df = pd.read_csv(file_path, delimiter=',', header='infer', names=None, skiprows=None)
        for index, row in connection_df.iterrows():
            connection_list.append(Connection(row['input'], row['output'], row['signal_power']))
        # Streaming the signal
        net.stream(connection_list, signal_power, label)

    return connection_list


def plot_traffic_matrix(traffic_matrix, transceiver_type, integer, M, MC):

    traffic_matrix_pd = pd.DataFrame(traffic_matrix, columns=['A', 'B', 'C', 'D', 'E', 'F'],
                                     index=['A', 'B', 'C', 'D', 'E', 'F'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(traffic_matrix_pd, annot=True, fmt='.3g', cmap="viridis", annot_kws={"size": 10})
    if integer == 0:
        plt.title('Initial Traffic Matrix for "' + transceiver_type + '"', **csfont_)
    else:
        plt.title('Traffic Matrix for "' + transceiver_type + '" case after deployment', **csfont_)
    plt.xlabel('Input Nodes', **csfont)
    plt.ylabel('Output Nodes', **csfont)
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')
    plt.tight_layout()

    if integer == 0:  # Initial matrix
        # Ensure base_path directory exists
        base_path = csv_switching_matrix
        os.makedirs(base_path, exist_ok=True)

        # Construct the full path for saving the plot
        log_file = fr'traffic_matrix_{M}.png'
        full_path = os.path.join(base_path, log_file)
        try:
            plt.savefig(full_path)
            # print(f"Plot saved successfully at: {full_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            # Clear the current figure to avoid overlap in future plots
            plt.clf()
            plt.close()
    elif integer == 1:  # Deployed matrix
        # Ensure base_path directory exists
        base_path = csv_switching_matrix
        os.makedirs(base_path, exist_ok=True)

        # Construct the full path for saving the plot
        log_file = fr'traffic_matrix_{M}_deploy{MC}.png'
        full_path = os.path.join(base_path, log_file)

        # Checking for error before to sabe the image
        try:
            plt.savefig(full_path)
            # print(f"Plot saved successfully at: {full_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            # Clear the current figure to avoid overlap in future plots
            plt.clf()
            plt.close()
    #plt.show()


# Compute the average latency and snr
def average_parameter(latency_array, snr_array):

    # flag values for latency -1, indicate rejected connections
    # That's why we only want the positives latencies in the array
    non_zero_elements = latency_array[latency_array > 0]
    mean_latency = np.mean(non_zero_elements)

    # flag values for snr -5, indicate rejected connections
    non_negative_elements = snr_array[snr_array > 0]
    mean_snr_linear = np.mean(db_to_linear(non_negative_elements))   # Convert to linear again before compute the mean
    mean_snr_db = linear_to_db(mean_snr_linear)

    return mean_latency, mean_snr_db


