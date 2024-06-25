# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from core.elements import csv_route_space, plt, csfont, csfont_
from main_congestion_scenario import MC
from core.parameters import M
import pandas as pd
import numpy as np
import seaborn as sns
import ast
import itertools
from matplotlib.ticker import FuncFormatter

option ='Fixed M'   # 'Congestion-scenario'
columns = [str(i) for i in range(1, MC)]

def dataf_generator(transceiver_type, type, flag):
    # verfiy this before to start, check if the root coincide with you root
    # FIRST CASE: FOR FIXED M SCENARIO
    if flag == 0:
        route = csv_route_space
    # SECOND CASE: FOR CONGESTION SCENARIO
    else:
        route = csv_route_space / transceiver_type

    df_snr = pd.DataFrame(columns = columns)
    df_latency = pd.DataFrame(columns= columns)
    df_bit_rate = pd.DataFrame(columns = columns)
    df_max_capacity = pd.DataFrame(columns= columns)
    df_min_capacity = pd.DataFrame(columns = columns)
    df_total_capacity = pd.DataFrame(columns = columns)
    df_blocking_events = pd.DataFrame(columns = columns)
    df_max_gsnr = pd.DataFrame(columns = columns)
    df_min_gsnr = pd.DataFrame(columns=columns)
    df_rejected_connections = pd.DataFrame(columns=columns)
    df_successful_connections = pd.DataFrame(columns=columns)

    # Add the first row of df_1 to df_final
    for i in range(1,MC):     #--->CHANGE IT ACCORDING THE NUMBER OF COLUMNS YOU HAVE IN THE DF

        # Read CSV file
        file_path = route /f'MC_{i}_{transceiver_type}.csv'
        df = pd.read_csv(file_path)
        df_snr = df_snr.append(df.iloc[0], ignore_index=True)
        df_latency = df_latency.append(df.iloc[1], ignore_index=True)
        df_bit_rate = df_bit_rate.append(df.iloc[2], ignore_index=True)
        df_max_capacity = df_max_capacity.append(df.iloc[3], ignore_index=True)
        df_min_capacity = df_min_capacity.append(df.iloc[4], ignore_index=True)
        df_total_capacity = df_total_capacity.append(df.iloc[5], ignore_index=True)
        df_blocking_events = df_blocking_events.append(df.iloc[6], ignore_index=True)
        df_max_gsnr = df_max_gsnr.append(df.iloc[7], ignore_index=True)
        df_min_gsnr = df_min_gsnr.append(df.iloc[8], ignore_index=True)
        df_rejected_connections = df_rejected_connections.append(df.iloc[9], ignore_index=True)
        df_successful_connections = df_successful_connections.append(df.iloc[10], ignore_index=True)

    if type == 'capacity':
        mean_values = df_total_capacity.mean(axis=1)   # to compute it per row
    elif type == 'snr':
        mean_values = df_snr.mean(axis=1)
    elif type =='Latency':
        mean_values = df_latency.mean(axis=1)
    elif type =='avg_capacity':
        mean_values = df_bit_rate.mean(axis=1)
    elif type == 'max_capacity':
        mean_values = df_max_capacity.mean(axis=1)
    elif type == 'min_capacity':
        mean_values = df_min_capacity.mean(axis=1)
    elif type == 'blocking_events':
        mean_values = df_blocking_events.mean(axis=1)
    elif type == 'max_gsnr':
        mean_values = df_max_gsnr.mean(axis=1)
    elif type == 'min_gsnr':
        mean_values = df_min_gsnr.mean(axis=1)
    elif type == 'rejected_connections':
        mean_values = df_rejected_connections.mean(axis=1)
    elif type == 'accepted_connections':
        mean_values = df_successful_connections.mean(axis=1)
    else:
        print('Error type')

    filename = csv_route_space / f'total_results_{type}.csv'
    if filename.exists():
        total_df = pd.read_csv(filename)
    else:
        total_df = pd.DataFrame(columns=['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate'])

        # Ensure the total_df has the correct number of rows to match the new data
    if len(total_df) < len(mean_values):
        total_df = total_df.reindex(range(len(mean_values)))

        # Append the new data to the appropriate column in the total_df DataFrame
    total_df[transceiver_type] = mean_values
    # Save the DataFrame to a CSV file
    total_df.to_csv(filename, index=False)

    print(total_df)


# Define a function to convert arrow notation to concatenated strings
def convert_arrows_to_concatenated(arrows):
    concatenated_list = []
    for arrow in arrows:
        # Split each arrow notation into source and target
        source, target = arrow.split('->')
        # Concatenate source and target without arrow symbol
        concatenated = source + target
        # Append to the result list
        concatenated_list.append(concatenated)
    return concatenated_list


def convert_into_df(type):
    #filename = csv_route_space/'Congestion_scenario'/'not_full_network'/f'total_results_{type}.csv
    filename = csv_route_space/'Fixed_M'/'Results'/'not_full_network results'/f'total_results_{type}.csv'
    df_all_transceiver = pd.read_csv(filename)
    return (df_all_transceiver)


def gbps_formatter(x, pos):
    """The two args are the value and tick position."""
    return f'{x / 1e9:.2f} Gbps'


def graph_generator(data, type, csfont, csfont_, M):
    # GRAPH -----------------------------------------------------------------------------------------------------------
    m = columns
    if type == 'capacity' or type == 'avg_capacity' or type == 'min_capacity' or type=='max_capacity':

        if type == 'capacity':  # if type == 'avg_capacity':
            # Set up Seaborn style
            sns.set(style="darkgrid")
            plt.plot(m, data['Fixed-Rate'], color='red', marker='o', linestyle='-', linewidth=2,label='Fixed Rate')
            plt.plot(m, data['Flex-Rate'], color='orange', marker='o', linestyle='-', linewidth=2, label='Flex Rate')
            plt.plot(m, data['Shannon-Rate'], color='blue', marker='o', linestyle='-', linewidth=2, label='Shannon Rate')
            plt.title(f'Total Capacity per each M value', **csfont_)

        else:
            # Add 'm' as a column to the DataFrame
            data['m'] = m
            # Melt the data to long format
            data_long = pd.melt(data, id_vars=['m'], value_vars=['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate'],
                                var_name='Rate Type', value_name='Value')
            sns.set(style="darkgrid")
            # Create a bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='m', y='Value', hue='Rate Type', data=data_long, palette=['red', 'orange', 'blue'])
            if type == 'avg_capacity':
                plt.title(f'Average capacity per each M value', **csfont_)
            if type == 'max_capacity':
                plt.title(f'Average Maximum Capacity per each M value', **csfont_)
            if type == 'min_capacity':
                plt.title(f'Average Minimum Capacity per each M value', **csfont_)

        # Adding labels to the axes
        plt.xlabel('M', **csfont)
        plt.ylabel('Capacity [Gbps]', **csfont)
        # Manually format the y-axis tick labels
        # plt.gca().set_yticklabels([f'{int(y)}e9' for y in plt.gca().get_yticks()])
        plt.gca().set_xticklabels([f'M={int(x + 1)}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.xticks(fontsize=9, fontname='Times New Roman')
        plt.yticks(fontsize=9, fontname='Times New Roman')
        plt.show()

    elif type == 'snr' or type == 'max_gsnr' or type == 'min_gsnr':
        # Add 'm' as a column to the DataFrame
        data['m'] = m
        # Melt the data to long format
        data_long = pd.melt(data, id_vars=['m'], value_vars=['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate'],
                            var_name='Rate Type', value_name='Value')
        sns.set(style="darkgrid")
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='m', y='Value', hue='Rate Type', data=data_long, palette=['red', 'orange', 'blue'])

        # Adding labels to the axes
        plt.xlabel('M', **csfont)
        plt.ylabel('SNR [dB]', **csfont)
        if type == 'snr':
            plt.title(f'Average SNR per each M value', **csfont_)
        if type == 'max_gsnr':
            plt.title(f'Average Maximum SNR per each M value', **csfont_)
        if type == 'min_gsnr':
            plt.title(f'Average Minimum SNR per each M value', **csfont_)
        # Manually format the y-axis tick labels
        # plt.gca().set_yticklabels([f'{int(y)}e9' for y in plt.gca().get_yticks()])
        plt.gca().set_xticklabels([f'M={int((x + 1))}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.xticks(fontsize=9, fontname='Times New Roman')
        plt.yticks(fontsize=9, fontname='Times New Roman')
        plt.show()

    elif type == 'Latency':
        # Add 'm' as a column to the DataFrame
        data['m'] = m
        # Melt the data to long format
        data_long = pd.melt(data, id_vars=['m'], value_vars=['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate'],
                            var_name='Rate Type', value_name='Value')
        sns.set(style="darkgrid")
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='m', y='Value', hue='Rate Type', data=data_long, palette=['red', 'orange', 'blue'])

        # Adding labels to the axes
        plt.xlabel('M', **csfont)
        plt.ylabel('Latency [ms]', **csfont)
        plt.title(f'Average Latency obtained according each M value', **csfont_)
        # Manually format the y-axis tick labels
        # plt.gca().set_yticklabels([f'{int(y)}e9' for y in plt.gca().get_yticks()])
        plt.gca().set_xticklabels([f'M={int(x + 1)}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.xticks(fontsize=9, fontname='Times New Roman')
        plt.yticks(fontsize=9, fontname='Times New Roman')
        plt.show()

    elif  type == 'rejected_connections' or type =='accepted_connections': #--------------------------------------------
        # Add 'm' as a column to the DataFrame
        data['m'] = m
        # Melt the data to long format
        data_long = pd.melt(data, id_vars=['m'], value_vars=['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate'],
                            var_name='Rate Type', value_name='Value')
        sns.set(style="darkgrid")
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='m', y='Value', hue='Rate Type', data=data_long, palette=['red', 'orange', 'blue'])

        # Adding labels to the axes
        plt.xlabel('M', **csfont)
        plt.ylabel('Number of Connections', **csfont)
        if type == 'rejected_connections':  #-------------------------------------------------------------
            plt.title(f'Average Rejected Connections per each M values, MC=10', **csfont_)

        if type == 'accepted_connections':  # -------------------------------------------------------------
            plt.title(f'Average Accepted Connections  per each M values, MC=10', **csfont_)
        # Manually format the y-axis tick labels
        # plt.gca().set_yticklabels([f'{int(y)}e9' for y in plt.gca().get_yticks()])
        plt.gca().set_xticklabels([f'M={int(x + 1)}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.xticks(fontsize=9, fontname='Times New Roman')
        plt.yticks(fontsize=9, fontname='Times New Roman')
        plt.show()


    elif type=='blocking_events':  #-----------------------------------------------------------------------------------
        # Set up Seaborn style
        sns.set(style="darkgrid")
        plt.plot(m, data['Fixed-Rate'], color='red', marker='.', linestyle='--', label='Fixed Rate')
        plt.plot(m, data['Flex-Rate'], color='orange', marker='.', linestyle='--', label='Flex Rate')
        plt.plot(m, data['Shannon-Rate'], color='blue', marker='.', linestyle='--', label='Shannon Rate')
        # Adding labels to the axes
        plt.xlabel('M', **csfont)
        plt.ylabel('Blocking Events', **csfont)
        plt.title(f'Average Number of Blocking Events per M, MC=10 runs', **csfont_)
        # Manually format the y-axis tick labels
        # plt.gca().set_yticklabels([f'{int(y)}e9' for y in plt.gca().get_yticks()])
        plt.gca().set_xticklabels([f'M={int(x + 1)}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.xticks(fontsize=9, fontname='Times New Roman')
        plt.yticks(fontsize=9, fontname='Times New Roman')
        plt.show()

    elif type == 'capacity_vs_line':
        # Plotting the histograms
        bar_width = 1.5
        index = convert_arrows_to_concatenated(data.index)
        spacing = 1  # Spacing between groups

        # Plotting with Seaborn
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))
        # Calculate positions for each bar group
        # Calculate positions for each bar group with added group spacing
        bar1 = [i * (3 * bar_width + spacing) for i in range(len(index))]
        bar2 = [i + bar_width for i in bar1]
        bar3 = [i + bar_width for i in bar2]

        plt.bar(bar1, data['Fixed-Rate'], color='red', width=bar_width, label='Fixed Rate')
        plt.bar(bar2, data['Flex-Rate'], color='orange', width=bar_width, label='Flex Rate')
        plt.bar(bar3, data['Shannon-Rate'], color='blue', width=bar_width, label='Shannon Rate')

        # Adding labels to the axes
        csfont = {'fontname': 'Times New Roman'}
        csfont_ = {'fontname': 'Times New Roman', 'fontsize': 14}

        plt.xlabel('Lines')
        plt.ylabel('Capacity [Gbps]', **csfont)
        plt.title(f'Per-link Average Capacity for M={M}', **csfont_)

        # Format the y-axis to show values in Gbps
        formatter = FuncFormatter(gbps_formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        # Set x-ticks to the index values
        # Calculate tick positions as the middle of each group
        tick_positions = [r + bar_width for r in bar1]
        plt.xticks(tick_positions, index, fontsize=9, fontname='Times New Roman')
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.show()

    elif type == 'spectral':
        # Plotting the histograms
        bar_width = 2
        index = convert_arrows_to_concatenated(data.index)
        spacing = 1.5  # Spacing between groups

        # Plotting with Seaborn
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))
        # Calculate positions for each bar group
        # Calculate positions for each bar group with added group spacing
        bar1 = [i * (3 * bar_width + spacing) for i in range(len(index))]
        bar2 = [i + bar_width for i in bar1]
        bar3 = [i + bar_width for i in bar2]

        plt.bar(bar1, data['Fixed-Rate'], color='r', width=bar_width, label='Fixed Rate')
        plt.bar(bar2, data['Flex-Rate'], color='orange', width=bar_width, label='Flex Rate')
        plt.bar(bar3, data['Shannon-Rate'], color='b', width=bar_width, label='Shannon Rate')

        # Adding labels to the axes
        csfont = {'fontname': 'Times New Roman'}
        csfont_ = {'fontname': 'Times New Roman', 'fontsize': 14}

        plt.xlabel('Lines')
        plt.ylabel('Spectral Occupation %', **csfont)
        plt.title(f'Spectral Occupation per Line for M={M}', **csfont_)

        # Set x-ticks to the index values
        # Calculate tick positions as the middle of each group
        tick_positions = [r + bar_width for r in bar1]
        plt.xticks(tick_positions, index, fontsize=9, fontname='Times New Roman')
        # Set the y-axis limit to a maximum of 100
        plt.ylim(0, 100)
        # Set y-ticks, ensuring the maximum is 100
        yticks = np.arange(0, 101, 10)
        plt.yticks(yticks)

        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.show()


 # Function to convert string representation of list into actual list
def str_to_list(s):
    return ast.literal_eval(s)


def combinations_node():
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    # Generate all possible combinations in the format "A->B"
    combinations = [f"{a}->{b}" for a, b in itertools.permutations(nodes, 2)]
    return combinations


def generate_graph_per_line(transceiver_list, congestion ,fixed_M):
    # List of nodes
    combinations = combinations_node()
    capacity_vs_line = pd.DataFrame(index = combinations)

    # verfiy this before to start, check if the root coincide with you root
    for i in transceiver_list:
        m = 4
        if congestion == 1:
            route = csv_route_space / i / f'MC_{m}_lines_{i}.csv'
        elif fixed_M == 1:
            route = csv_route_space /'Fixed_M'/f'MC_{m}_lines_{i}.csv'
        else:
            print('Set the flag of congestion of fixed_M')
        df_aux = pd.read_csv(route)

        if df_aux['paths'].nunique() == 1:
            print(f"All elements in column are the same.")
        else:
            print(f"Not all elements in column not are the same.")

        # Convert the 'paths' column of the first row into a list
        paths_list = str_to_list(df_aux.loc[0, 'paths'])

        # Extract bit_rate lists and convert them from string to actual lists
        bit_rate_lists = df_aux['bit_rate'].apply(str_to_list).tolist()

        # Create a dictionary to store the summed values and counts for averaging
        bit_rate_sums = {}
        bit_rate_counts = {}
        for path, rates in zip(paths_list, zip(*bit_rate_lists)):
            if path not in bit_rate_sums:
                bit_rate_sums[path] = sum(rates)
                bit_rate_counts[path] = len(rates)
            else:
                bit_rate_sums[path] += sum(rates)
                bit_rate_counts[path] += len(rates)

        # Calculate the averaged bit rates
        averaged_bit_rates = {path: bit_rate_sums[path] / bit_rate_counts[path] for path in bit_rate_sums}

        # Create a DataFrame from the averaged bit rates
        avg_df = pd.DataFrame(list(averaged_bit_rates.values()), index=list(averaged_bit_rates.keys()), columns=[i])
        # Add the mean values to the capacity_vs_line DataFrame
        capacity_vs_line[i] = avg_df.reindex(capacity_vs_line.index).fillna(0)
    if congestion == 1:
        graph_generator(capacity_vs_line, 'capacity_vs_line', csfont, csfont_, m)
    if fixed_M == 1:
        graph_generator_fixedM(capacity_vs_line, 'capacity_vs_line', csfont, csfont_)


# Function to count zeros in the 10 columns
def count_zeros(row):
    return row[1:].astype(int).sum()


def occupation_graph(type, m, number_channels):
    combinations = combinations_node()
    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=combinations)

    for i in transceiver_list:
        if type == 'congestion' :
            route = csv_route_space / i / f'M={m}'/'route_space.csv'
            df_aux = pd.read_csv(route)
        if type == 'fixed_M':
            route = csv_route_space /'Fixed_M'/i / 'route_space.csv'
            df_aux = pd.read_csv(route)

        # Simplify paths to direct connections and count zeros
        df_aux['Simplified_Path'] = df_aux['Paths'].apply(lambda x: '->'.join([x.split('->')[0], x.split('->')[-1]]))
        df_aux['Zero_Count'] = (df_aux.iloc[:, 1:] == 0).sum(axis=1)

        # Aggregate and calculate average zeros for each combination
        for combo in combinations:
            avg_zeros = df_aux[df_aux['Simplified_Path'] == combo]['Zero_Count'].mean()
            result_df.at[combo, f'{i}'] = (avg_zeros / number_channels) * 100 if pd.notna(avg_zeros) else 0
    if type == 'congestion':
        graph_generator(result_df, 'spectral', csfont,csfont_, m)
    if type == 'fixed_M':
        graph_generator_fixedM(result_df, 'spectral', csfont, csfont_)
    print()


def graph_generator_fixedM(data, type, csfont, csfont_):
    # GRAPH ----------------------------------------------------------------------------------------------------------
    if type == 'capacity' or type == 'avg_capacity' or type == 'max_capacity' or type == 'min_capacity':
        # Setting the style for the plot
        sns.set(style="darkgrid")
        #df_gbps = data / 1e3
        # Transpose the DataFrame for easier plotting
        df_t = data.T
        df_t.columns = ['Capacity']

        # Plot the data
        plt.figure(figsize=(8, 6))
        bars = plt.bar(df_t.index, df_t['Capacity'], color=[ 'red', 'orange','blue'])
        plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Capacity [Gbps]', fontsize=12, fontname='Times New Roman')
        if type =='capacity':
            plt.title('Total Capacity per Transceiver with M = 4, 50 Runs', fontsize=14, fontname='Times New Roman')

        elif type =='avg_capacity':
            plt.title('Average Capacity per Transceiver with M = 4, 50 Runs', fontsize=14, fontname='Times New Roman')

        elif type=='max_capacity':
            plt.title('Average Maximum Capacity per Transceiver with M = 4, 50 Runs', fontsize=14, fontname='Times New Roman')
            # Adding the exact number on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.12f}', ha='center', va='bottom',
                         fontsize=10, fontname='Times New Roman')

        elif type == 'min_capacity':
            plt.title('Average Minimum Capacity per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')
            # Adding the exact number on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.12f}', ha='center', va='bottom',
                         fontsize=10, fontname='Times New Roman')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.show()

    elif type == 'snr' or type == 'max_gsnr' or type == 'min_gsnr': #---------------------------------------------------
        # Setting the style for the plot
        sns.set(style="darkgrid")
        # df_gbps = data / 1e3
        # Transpose the DataFrame for easier plotting
        df_t = data.T
        df_t.columns = ['SNR']

        # Plot the data
        plt.figure(figsize=(8, 6))
        bars = plt.bar(df_t.index, df_t['SNR'], color=['red', 'orange', 'blue'])
        plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Signal to Noise Ratio [dB]', fontsize=12, fontname='Times New Roman')
        if type == 'snr':
            plt.title('Average SNR per Transceiver with M = 4, 50 Runs', fontsize=14, fontname='Times New Roman')
        if type == 'max_gsnr':
            plt.title('Average of Maximum SNR per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')
        if type == 'min_gsnr':
            plt.title('Average of Minimum SNR per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        # Adding the exact number on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.12f}', ha='center', va='bottom',
                     fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.show()

    elif type == 'Latency': #------------------------------------------------------------------------------------------
        # Setting the style for the plot
        sns.set(style="darkgrid")
        # df_gbps = data / 1e3
        # Transpose the DataFrame for easier plotting
        df_t = data.T
        df_t.columns = ['Latency']

        # Plot the data
        plt.figure(figsize=(8, 6))
        bars = plt.bar(df_t.index, df_t['Latency'], color=['red', 'orange', 'blue'])
        plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Latency [ms]', fontsize=12, fontname='Times New Roman')
        plt.title('Average Latency per Transceiver with M = 4, 50 Runs', fontsize=14, fontname='Times New Roman')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.12f}', ha='center', va='bottom',
                     fontsize=10, fontname='Times New Roman', fontweight='normal')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.show()

    elif type == 'blocking_events' or type == 'accepted_connections' or type == 'rejected_connections': #--------------
        # Set up Seaborn style
        # Setting the style for the plot
        sns.set(style="darkgrid")

        # Plot the data
        plt.figure(figsize=(8, 6))
        if type == 'blocking_events':
            # Transpose the DataFrame for easier plotting
            df_t = data.T
            df_t.columns = ['blocking_event']
            bars = plt.bar(df_t.index, df_t['blocking_event'], color=['red', 'orange', 'blue'])
            plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
            plt.ylabel('Blocking Events', fontsize=12, fontname='Times New Roman')
            plt.title('Average Blocking Event Numbers per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')

        if type == 'accepted_connections':
            # Transpose the DataFrame for easier plotting
            df_t = data.T
            df_t.columns = ['accepted_connections']
            bars = plt.bar(df_t.index, df_t['accepted_connections'], color=['red', 'orange', 'blue'])
            plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
            plt.ylabel('Accepted Connections', fontsize=12, fontname='Times New Roman')
            plt.title('Average Accepted Connections per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')

        if type == 'rejected_connections':
            # Transpose the DataFrame for easier plotting
            df_t = data.T
            df_t.columns = ['rejected_connections']
            bars = plt.bar(df_t.index, df_t['rejected_connections'], color=['red', 'orange', 'blue'])
            plt.xlabel('Transceiver Types', fontsize=12, fontname='Times New Roman')
            plt.ylabel('Rejected Connections', fontsize=12, fontname='Times New Roman')
            plt.title('Average Rejected Connections per Transceiver with M = 4, 50 Runs',
                      fontsize=14, fontname='Times New Roman')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom',
                     fontsize=10, fontname='Times New Roman', fontweight='normal')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.show()

    elif type == 'capacity_vs_line': #---------------------------------------------------------------------------------
        # Plotting the histograms
        bar_width = 1.5
        index = convert_arrows_to_concatenated(data.index)
        spacing = 1  # Spacing between groups

        # Plotting with Seaborn
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))
        # Calculate positions for each bar group
        # Calculate positions for each bar group with added group spacing
        bar1 = [i * (3 * bar_width + spacing) for i in range(len(index))]
        bar2 = [i + bar_width for i in bar1]
        bar3 = [i + bar_width for i in bar2]

        plt.bar(bar1, data['Fixed-Rate'], color='red', width=bar_width, label='Fixed Rate')
        plt.bar(bar2, data['Flex-Rate'], color='orange', width=bar_width, label='Flex Rate')
        plt.bar(bar3, data['Shannon-Rate'], color='blue', width=bar_width, label='Shannon Rate')

        # Adding labels to the axes
        csfont = {'fontname': 'Times New Roman'}
        csfont_ = {'fontname': 'Times New Roman', 'fontsize': 14}
        plt.xlabel('Lines')
        plt.ylabel('Capacity [Gbps]', **csfont)
        plt.title(f'Per-link Average Capacity for M={4}, with MC={MC-1}', **csfont_)

        # Format the y-axis to show values in Gbps
        formatter = FuncFormatter(gbps_formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        # Set x-ticks to the index values
        # Calculate tick positions as the middle of each group
        tick_positions = [r + bar_width for r in bar1]
        plt.xticks(tick_positions, index, fontsize=9, fontname='Times New Roman')
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.show()

    elif type == 'spectral': # ---------------------------------------------------------------------------------------
        # Plotting the histograms
        bar_width = 2
        index = convert_arrows_to_concatenated(data.index)
        spacing = 1.5  # Spacing between groups

        # Plotting with Seaborn
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))
        # Calculate positions for each bar group
        # Calculate positions for each bar group with added group spacing
        bar1 = [i * (3 * bar_width + spacing) for i in range(len(index))]
        bar2 = [i + bar_width for i in bar1]
        bar3 = [i + bar_width for i in bar2]

        plt.bar(bar1, data['Fixed-Rate'], color='r', width=bar_width, label='Fixed Rate')
        plt.bar(bar2, data['Flex-Rate'], color='orange', width=bar_width, label='Flex Rate')
        plt.bar(bar3, data['Shannon-Rate'], color='b', width=bar_width, label='Shannon Rate')

        # Adding labels to the axes
        csfont = {'fontname': 'Times New Roman'}
        csfont_ = {'fontname': 'Times New Roman', 'fontsize': 14}

        plt.xlabel('Lines')
        plt.ylabel('Spectral Occupation %', **csfont)
        plt.title(f'Single Matrix Scenario - Spectral occupation per Line for M={4}', **csfont_)

        # Set x-ticks to the index values
        # Calculate tick positions as the middle of each group
        tick_positions = [r + bar_width for r in bar1]
        plt.xticks(tick_positions, index, fontsize=9, fontname='Times New Roman')
        plt.tight_layout()
        plt.legend(loc='best', fontsize=10)
        plt.show()


# CHOOSE THE TYPE OF DF YOU WANT TO CREATE IN DATAF_GENERATOR()
type = 'capacity'
#'avg_capacity ','snr', 'capacity', 'Latency', 'blocking_events', 'max_capacity', 'min_capacity', 'max_gsnr',
# 'min_gsnr', 'rejected_connections', 'accepted_connections'
transceiver_list = ['Fixed-Rate', 'Flex-Rate', 'Shannon-Rate']

# READ THE INFO OF EACH METRIC FOR EACH TRANSCEIVER AND PUT ALL OF THEM IN A DF
# Run only once
# for i in transceiver_list:
#    dataf_generator(i, type, 1)     # FLAG=1 CONGESTION SCENARIO

# THIS FUNCTION RECEIVE THE TYPE OF GRAPH YOU WANT TO SEE---------------------------------
# OPTIONS : capacity, snr, latency, bit_rate, blocking-events  (IN LOWERCASE)
#graph_generator(convert_into_df(type), type, csfont, csfont_, 10)
#graph_generator_fixedM(convert_into_df(type), type, csfont, csfont_)


# Graph per line or per link
# First integer : Activate the graph for congestion scenario with Flag = 1
# Second Integer = Activate the graph for the fixed M scenario with Flag = 1
#generate_graph_per_line(transceiver_list, congestion=0, fixed_M=1)

# Choose options : 'congestion' or 'fixed_M'  (RESPECT THE CAPITAL AND LOWER LETTERS)
# AND ADDNUMBER OF M, number_channels= 6 or 10 depending on the case
# for full_network or not full_network --> ('Congestion', 10, 6)
#occupation_graph('congestion', 10, 6)
