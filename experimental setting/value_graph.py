import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
from itertools import product
import dataframe_image as dfi
import numpy as np
from scipy import integrate

# Choose the DPI of the graphs you want to print
DPI = 300



# List of models you want to create graphs of
MODELS=["LinearDiscriminantAnalysis", "LogisticRegression", "RandomForestClassifier", "VotingClassifier", "XGBClassifier"]

def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)

# Creates a value table
def create_table(df, k_fn, save_directory, name):
    # create pivot table
    pivot_df = df[(df['k_fn'] == k_fn)]
    pivot_df = pivot_df[["dataset", "calibrator", "value", "value_optimal", "model"]]
    pivot_df = pivot_df.rename(columns={'value': 'theoretical', 'value_optimal': 'empirical'})
    pivot_df = pivot_df.pivot_table(index='model', columns=['dataset', 'calibrator'], values=['theoretical', 'empirical'])
    pivot_df = pivot_df.stack(level=0, future_stack=True)

    # Initialize an empty DataFrame with the same structure as pivot_df to store the normalized values
    normalized_pivot_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    
    # Iterate over each index level in the pivot table
    for model in pivot_df.index.get_level_values('model').unique():
        # Filter the pivot_df for the current model
        model_df = pivot_df.loc[model]
        # Normalize the values for each column in the current model
        for column in model_df.columns.get_level_values(0).unique():
            normalized_values = (model_df[column] - model_df[column].min().min()) / (model_df[column].max().max() - model_df[column].min().min())
            # Assign the normalized values to the corresponding column in the normalized DataFrame
            normalized_pivot_df.loc[model, column] = normalized_values.to_numpy()

    pd.set_option('future.no_silent_downcasting', True)
    normalized_pivot_df=normalized_pivot_df.fillna(1)
    # apply gradient
    cmap = LinearSegmentedColormap.from_list("red_to_green", ["#FF3333","white", "green"],N=65536,gamma=1)
    styled_df = pivot_df.style.background_gradient(cmap=cmap,axis=None,gmap=normalized_pivot_df)
    for dataset_tag in pivot_df.columns.get_level_values(0).unique():
        styled_df.highlight_max(subset=dataset_tag, props='color: black;  font-weight: bold;', axis=1)

    title = f"{name} Values Table, k_fp {df["k_fp"].max()}, k_fn {k_fn}, it goes from red (worst) to green (better) with the bold ones being the best on that model and dataset"
    styled_df = styled_df.set_caption(title)
    styled_df = styled_df.set_properties(**{'color': 'black'})
    styled_df = styled_df.set_table_styles([
        {'selector': 'td', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'td', 'props': [('border-bottom', '1px solid black'),('padding', '2px')]},
        {'selector': 'th', 'props': [('font-size', '12px'), ('width', '65px')]},
        {'selector': 'th', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(7)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(12)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(17)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(22)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'caption', 'props': [('font-size', '20px bold'),
                                          ('font-weight', 'bold'), ('padding', '5px')]},
    ])
    for column in pivot_df.columns.get_level_values('dataset').unique():
        styled_df = styled_df.set_table_styles({(column, "pla"): [{'selector': 'td', 'props': 'border-right: 2px solid black'}]}, overwrite=False, axis=0)
    for l0 in pivot_df.index.get_level_values('model').unique():
        styled_df.set_table_styles({(l0, 'theoretical'): [{'selector': '', 'props': 'border-bottom: 2px solid black;'}],
                        (l0, 'empirical'): [{'selector': '.level0', 'props': 'border-bottom: 2px solid black'}]},
                      overwrite=False, axis=1)
    dfi.export(styled_df, os.path.join(save_directory, f'Table_{k_fn}.png'), dpi=DPI)

# Creates a plot of the results, includes AUV in the labels
def plot_lines(final_df, dataset, model, k_fp, save_directory, start, finish, name):
    # Filter the DataFrame based on dataset, model, and matching k_fp and k_fn
    filtered_df = final_df[(final_df['dataset'] == dataset) & (final_df['model'] == model)]
    
    # Create a figure and a grid of subplots with 3 rows
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 12))
    
    fig.suptitle(f"{name} Values of {model} over dataset {dataset} with k_fp {k_fp}")

    # Define major ticks every 50 units
    major_ticks = np.arange(start-10, finish+0.1, 50)

    # Apply major ticks and grid to all subplots
    for ax in [ax1, ax2]:
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlim(start-10, finish+10)

    # Plot 'value' on the first subplot
    for calibrator in filtered_df['calibrator'].unique():
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]
        calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
        
        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_values = calibrator_data['value']
        area = integrate.simpson(value_values, x=k_fn_values) / (finish-start+1)
        
        label = f"{calibrator} AUV={area:.4f}"
        ax1.plot(k_fn_values, value_values, label=label)
    
    ax1.set_ylabel('theoretical value')
    ax1.legend()
    
    # Plot 'value_optimal' on the second subplot
    for calibrator in filtered_df['calibrator'].unique():
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]
        calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
        
        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_optimal_values = calibrator_data['value_optimal']
        area = integrate.simpson(value_optimal_values, x=k_fn_values) / (finish-start+1)
        
        label = f"{calibrator} AUV={area:.4f}"
        ax2.plot(k_fn_values, value_optimal_values, label=label)
    
    ax2.set_ylabel('empirical value')
    ax2.set_xlabel('k_fn')
    ax2.legend()
    
    # Adjust the layout to ensure the subplots do not overlap
    plt.tight_layout()
    # Save the figure to the specified directory
    
    if start == filtered_df['k_fn'].min() and finish == filtered_df['k_fn'].max():
        plt.savefig(os.path.join(save_directory, f'{dataset}_{model}_plot.png'), format="png", dpi=DPI)
    else:
        save_directory = os.path.join(save_directory, f"{start}_{finish}")
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(os.path.join(save_directory, f'{dataset}_{model}_plot.png'), format="png", dpi=DPI)
    plt.close()

# Creates a plot of the results with rejection, includes AUV in the labels
def rejection_lines(final_df, dataset, model, k_fp, save_directory, start, finish, name):
    # Filter the DataFrame based on dataset, model, and matching k_fp and k_fn
    filtered_df = final_df[(final_df['dataset'] == dataset) & (final_df['model'] == model)]
    
    # Create a figure and a grid of subplots with 3 rows
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 12))
    
    fig.suptitle(f"{name} Values of {model} over dataset {dataset} with k_fp {k_fp}")

    # Define major ticks every 50 units
    major_ticks = np.arange(start-10, finish+0.1, 50)

    # Apply major ticks and grid to all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlim(start-10, finish+10)

    # Plot 'value' on the first subplot
    for calibrator in filtered_df['calibrator'].unique():
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]
        calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
        
        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_values = calibrator_data['value']
        area = integrate.simpson(value_values, x=k_fn_values) / (finish-start+1)
        
        label = f"{calibrator} AUV={area:.4f}"
        ax1.plot(k_fn_values, value_values, label=label)
    
    ax1.set_ylabel('theoretical value')
    ax1.legend()
    
    # Plot 'value_optimal' on the second subplot
    for calibrator in filtered_df['calibrator'].unique():
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]
        calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
        
        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_optimal_values = calibrator_data['value_optimal']
        area = integrate.simpson(value_optimal_values, x=k_fn_values) / (finish-start+1)
        
        label = f"{calibrator} AUV={area:.4f}"
        ax2.plot(k_fn_values, value_optimal_values, label=label)
    
    ax2.set_ylabel('empirical value')
    ax2.legend()
    
    # Plot 'value_no_rej' on the third subplot
    for calibrator in filtered_df['calibrator'].unique():
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]
        calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_rejection_values = calibrator_data['value_no_rej']
        area = integrate.simpson(value_rejection_values, x=k_fn_values) / (finish-start+1)

        label = f"{calibrator} AUV={area:.4f}"
        ax3.plot(k_fn_values, value_rejection_values, label=label)

    ax3.set_xlabel('k_fn')
    ax3.set_ylabel('no rejection value')
    ax3.legend()

    # Adjust the layout to ensure the subplots do not overlap
    plt.tight_layout()
    # Save the figure to the specified directory
    
    if start == filtered_df['k_fn'].min() and finish == filtered_df['k_fn'].max():
        plt.savefig(os.path.join(save_directory, f'{dataset}_{model}_plot.png'), format="png", dpi=DPI)
    else:
        save_directory = os.path.join(save_directory, f"{start}_{finish}")
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(os.path.join(save_directory, f'{dataset}_{model}_plot.png'), format="png", dpi=DPI)
    plt.close()

# Creates a plot that compares the results among different runs of a calibrator
def plot_lines_by_run(final_dataframes_by_run, dataset, model, k_fp, calibrator, save_directory):
    # Create a figure and a grid of subplots with 2 rows
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 12))
    
    fig.suptitle(f"Values of {model} over dataset {dataset} with k_fp {k_fp} for calibrator {calibrator}")

    # Iterate over each run in the dictionary
    for run_key, final_df in final_dataframes_by_run.items():
        start=final_df['k_fn'].min()
        finish=final_df['k_fn'].max()
        # Filter the DataFrame based on dataset, model, and matching k_fp
        filtered_df = final_df[(final_df['dataset'] == dataset) & (final_df['model'] == model)]
        
        calibrator_data = filtered_df[filtered_df['calibrator'] == calibrator]

        # Calculate the area under the curve (AUV)
        k_fn_values = calibrator_data['k_fn']
        value_values = calibrator_data['value']
        value_values_optimal = calibrator_data['value_optimal']
        area = integrate.simpson(value_values, x=k_fn_values) / (finish-start+1)
        area_optimal = integrate.simpson(value_values_optimal, x=k_fn_values) / (finish-start+1)
        
        # Plot 'value' on the second subplot
        label = f"{run_key} AUV={area:.4f}"
        ax1.plot(k_fn_values, value_values, label=label)
        
        # Plot 'value_optimal' on the second subplot
        label = f"{run_key} AUV={area_optimal:.4f}"
        ax2.plot(k_fn_values, value_values_optimal, label=label)

    
    ax1.set_ylabel('theoretical value')
    ax1.legend()
    
    ax2.set_ylabel('empirical value')
    ax2.set_xlabel('k_fn')
    ax2.legend()
    
    # Adjust the layout to ensure the subplots do not overlap
    plt.tight_layout()
    # Save the figure to the specified directory
    plt.savefig(os.path.join(save_directory, f'{dataset}_{model}_plot.png'), format="png", dpi=DPI)
    plt.close()

# Function to calculate AUV values
def calculate_auv(df, start, finish):
    auv_data = []
    for (dataset, model), model_df in df.groupby(['dataset', 'model']):
        for calibrator in model_df['calibrator'].unique():
            calibrator_data = model_df[model_df['calibrator'] == calibrator]
            calibrator_data = calibrator_data[(calibrator_data['k_fn'] >= start) & (calibrator_data['k_fn'] <= finish)]
            k_fn_values = calibrator_data['k_fn']
            value_values = calibrator_data['value']
            value_optimal_values = calibrator_data['value_optimal']
            auv_value = integrate.simpson(value_values, x=k_fn_values) / (finish-start+1)
            auv_value_optimal = integrate.simpson(value_optimal_values, x=k_fn_values) / (finish-start+1)
            auv_data.append({
                'dataset': dataset,
                'model': model,
                'calibrator': calibrator,
                'auv_theoretical': auv_value,
                'auv_empirical': auv_value_optimal
            })
    return pd.DataFrame(auv_data)

# Modified create_table function to generate AUV tables
def create_auv_table(df, start, finish, save_directory, name):
    auv_df = calculate_auv(df, start, finish)
    
    pivot_df = auv_df.pivot_table(index='model', columns=['dataset', 'calibrator'], values=['auv_theoretical', 'auv_empirical'])
    pivot_df = pivot_df.stack(level=0, future_stack=True)

    # Initialize an empty DataFrame with the same structure as pivot_df to store the normalized values
    normalized_pivot_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    
    # Iterate over each index level in the pivot table
    for model in pivot_df.index.get_level_values('model').unique():
        # Filter the pivot_df for the current model
        model_df = pivot_df.loc[model]
        # Normalize the values for each column in the current model
        for column in model_df.columns.get_level_values(0).unique():
            normalized_values = (model_df[column] - model_df[column].min().min()) / (model_df[column].max().max() - model_df[column].min().min())
            # Assign the normalized values to the corresponding column in the normalized DataFrame
            normalized_pivot_df.loc[model, column] = normalized_values.to_numpy()

    pd.set_option('future.no_silent_downcasting', True)
    normalized_pivot_df = normalized_pivot_df.fillna(1)
    # apply gradient
    cmap = LinearSegmentedColormap.from_list("red_to_green", ["#FF3333","white", "green"], N=65536, gamma=1)
    styled_df = pivot_df.style.background_gradient(cmap=cmap, axis=None, gmap=normalized_pivot_df)
    for dataset_tag in pivot_df.columns.get_level_values(0).unique():
        styled_df.highlight_max(subset=dataset_tag, props='color: black;  font-weight: bold;', axis=1)

    title = f"{name} AUV Values Table, k_fn range {start} to {finish}, it goes from red (worst) to green (better) with the bold ones being the best on that model and dataset"
    styled_df = styled_df.set_caption(title)
    styled_df = styled_df.set_properties(**{'color': 'black'})
    styled_df = styled_df.set_table_styles([
        {'selector': 'td', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'td', 'props': [('border-bottom', '1px solid black'),('padding', '2px')]},
        {'selector': 'th', 'props': [('font-size', '12px'), ('width', '65px')]},
        {'selector': 'th', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(7)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(12)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(17)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(22)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'caption', 'props': [('font-size', '20px bold'),
                                          ('font-weight', 'bold'), ('padding', '5px')]},
    ])
    for l0 in pivot_df.index.get_level_values('model').unique():
        styled_df.set_table_styles({(l0, 'theoretical'): [{'selector': '', 'props': 'border-bottom: 2px solid black;'}],
                        (l0, 'empirical'): [{'selector': '.level0', 'props': 'border-bottom: 2px solid black'}]},
                      overwrite=False, axis=1)
        
    for column in pivot_df.columns.get_level_values('dataset').unique():
        styled_df = styled_df.set_table_styles({(column, "pla"): [{'selector': 'td', 'props': 'border-right: 2px solid black'}]}, overwrite=False, axis=0)
    for l0 in pivot_df.index.get_level_values('model').unique():
        styled_df.set_table_styles({(l0, 'auv_theoretical'): [{'selector': '', 'props': 'border-top: 2px solid black;'}],
                                    (l0, 'auv_empirical'): [{'selector': '.level0', 'props': 'border-top: 2px solid black'}]},
                                    overwrite=False, axis=1)
    
    styled_df = styled_df.format(precision=6)
    styled_df.to_excel(os.path.join(save_directory, f'AUV_Table_{start}_{finish}.xlsx'))
    dfi.export(styled_df, os.path.join(save_directory, f'AUV_Table_{start}_{finish}.png'), dpi=DPI)



# Main -----------------------------------------------------------------

start_time=current_milli_time()
# Get the directory containing value output
directory = os.path.join(os.getcwd(), "value")

# Disctionary to store DataFrames
dataframes_by_run = {}

pattern = r"(.+?)\.csv_(\w+)_(\d+)_(\w+)_(\w+)\.csv"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Match the filename against the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the matched groups
            dataset, model, run_number, calibrator, _ = match.groups()
            if(model not in MODELS):
                continue
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)
            match=re.match(r"(\w+)_costBased",calibrator)
            if match:
                calibrator=match.group(1)
            else:
                calibrator="base"
            # Add the extracted parts as columns to the DataFrame
            df['dataset'] = dataset
            df['model'] = model
            df['calibrator'] = calibrator
             # Group DataFrames by test run number
            if run_number not in dataframes_by_run:
                dataframes_by_run[run_number] = []
            dataframes_by_run[run_number].append(df)

final_dataframes_by_run={}
for i in dataframes_by_run.keys():
    final_dataframes_by_run[i]=pd.concat(dataframes_by_run[i], ignore_index=True)

pd.options.mode.copy_on_write = True

avg_df = final_dataframes_by_run[list(final_dataframes_by_run.keys())[0]].copy()

avg_df["value"] = pd.to_numeric(avg_df["value"], errors='coerce')
avg_df["value_optimal"] = pd.to_numeric(avg_df["value_optimal"], errors='coerce')

# Loop through each run
for i in list(final_dataframes_by_run.keys())[1:]:
    
    # Read the CSV file into a DataFrame
    df = final_dataframes_by_run[i].copy()
    
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    df["value_optimal"] = pd.to_numeric(df["value_optimal"], errors='coerce')
    
    # Calculate the average of the columns across all files
    avg_df["value"] += df["value"].fillna(0)
    avg_df["value_optimal"] += df["value_optimal"].fillna(0)

avg_df["value"] = avg_df["value"] / len(final_dataframes_by_run.keys())
avg_df["value_optimal"] = avg_df["value_optimal"] / len(final_dataframes_by_run.keys())

unique_combinations = list(product(avg_df['dataset'].unique(), avg_df['model'].unique()))

save_directory=os.path.join(directory,"tables")
os.makedirs(save_directory,exist_ok=True)

# Save average run tables
create_table(avg_df,10,save_directory,"Average")
create_table(avg_df,50,save_directory,"Average")
create_table(avg_df,100,save_directory,"Average")
create_table(avg_df,1000,save_directory,"Average")

print("finished tables")

save_directory=os.path.join(directory,"plot_lines")

# Save average plot lines
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, avg_df['k_fn'].min(), avg_df['k_fn'].max(),"Average")
    plot_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, avg_df['k_fn'].min(), 100,"Average")
    plot_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, 100, 500,"Average")
    plot_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, 500, avg_df['k_fn'].max(),"Average")

print("finished plot")

save_directory=os.path.join(directory,"rejection_lines")

# Save average rejection lines
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    rejection_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, avg_df['k_fn'].min(), avg_df['k_fn'].max(),"Average")
    rejection_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, avg_df['k_fn'].min(), 100,"Average")
    rejection_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, 100, 500,"Average")
    rejection_lines(avg_df,dataset,model,df['k_fp'].max(), save_directory, 500, avg_df['k_fn'].max(),"Average")

print("finished rejection")

# Save comparison of calibration results
save_directory=os.path.join(directory,"platt_lines")
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines_by_run(final_dataframes_by_run, dataset, model, df['k_fp'].max(), "pla", save_directory)
print("finished platt")

save_directory=os.path.join(directory,"bbq_lines")
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines_by_run(final_dataframes_by_run, dataset, model, df['k_fp'].max(), "bbq", save_directory)
print("finished bbq")

save_directory=os.path.join(directory,"iso_lines")
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines_by_run(final_dataframes_by_run, dataset, model, df['k_fp'].max(), "iso", save_directory)
print("finished iso")

save_directory=os.path.join(directory,"his_lines")
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines_by_run(final_dataframes_by_run, dataset, model, df['k_fp'].max(), "his", save_directory)
print("finished his")

save_directory=os.path.join(directory,"base_lines")
os.makedirs(save_directory,exist_ok=True)
for dataset, model in unique_combinations:
    plot_lines_by_run(final_dataframes_by_run, dataset, model, df['k_fp'].max(), "base", save_directory)

print("finished base")

# Save average run AUV tables
save_directory = os.path.join(os.getcwd(), "value", "auv_tables")
os.makedirs(save_directory, exist_ok=True)
create_auv_table(avg_df, avg_df['k_fn'].min(), avg_df['k_fn'].max(), save_directory, "Average")
create_auv_table(avg_df, avg_df['k_fn'].min(), 100, save_directory, "Average")
create_auv_table(avg_df, 100, 500, save_directory, "Average")
create_auv_table(avg_df, 500, avg_df['k_fn'].max(), save_directory, "Average")

print("finished auv")

for i in list(final_dataframes_by_run.keys()):
    df = final_dataframes_by_run[i].copy()
    
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    df["value_optimal"] = pd.to_numeric(df["value_optimal"], errors='coerce')

    # Create save directories for the individual run
    run_save_directory = os.path.join(directory, i)
    os.makedirs(run_save_directory, exist_ok=True)
    
    run_plot_save_directory = os.path.join(run_save_directory, "plot")
    os.makedirs(run_plot_save_directory, exist_ok=True)
    
    # Save individual run tables
    create_table(df, 10, run_save_directory,i)
    create_table(df, 50, run_save_directory,i)
    create_table(df, 100, run_save_directory,i)
    create_table(df, 1000, run_save_directory,i)
    
    # Save individual run plots
    for dataset, model in unique_combinations:
        plot_lines(df, dataset, model, df['k_fp'].max(), run_plot_save_directory, df['k_fn'].min(), df['k_fn'].max(),i)
        plot_lines(df, dataset, model, df['k_fp'].max(), run_plot_save_directory, df['k_fn'].min(), 100,i)
        plot_lines(df, dataset, model, df['k_fp'].max(), run_plot_save_directory, 100, 500,i)
        plot_lines(df, dataset, model, df['k_fp'].max(), run_plot_save_directory, 500, df['k_fn'].max(),i)
        
    save_directory = os.path.join(run_save_directory, "auv_tables")
    os.makedirs(save_directory, exist_ok=True)
    create_auv_table(df, df['k_fn'].min(), df['k_fn'].max(), save_directory, i)
    create_auv_table(df, df['k_fn'].min(), 100, save_directory, i)
    create_auv_table(df, 100, 500, save_directory, i)
    create_auv_table(df, 500, df['k_fn'].max(), save_directory, i)
    print(i)

print("finished single")



print(current_milli_time()-start_time)

# save_directory=os.path.join(directory,"tables","sd")
# os.makedirs(save_directory,exist_ok=True)

# sd = avg_df.copy()
# # tmp=avg_df.copy()
# for i in list(final_dataframes_by_run.keys()):
#     sd['value']=sd['value'].add(avg_df['value'].sub(final_dataframes_by_run[i]['value']).pow(2))
#     sd['value_optimal']=sd['value_optimal'].add(avg_df['value_optimal'].sub(final_dataframes_by_run[i]['value_optimal']).pow(2))

# sd['value']= sd['value'].div(len(final_dataframes_by_run.keys()))
# sd['value_optimal']= sd['value_optimal'].div(len(final_dataframes_by_run.keys()))
#sd['value']= sd['value'].pow(0.5) bugged does not work
#sd['value_optimal']= sd['value_optimal'].pow(0.5)
# create_table(sd,10,save_directory)
# create_table(sd,50,save_directory)
# create_table(sd,100,save_directory)
# create_table(sd,1000,save_directory)

# for i in list(final_dataframes_by_run.keys()):

#     save_directory=os.path.join(os.getcwd(),i)
#     os.makedirs(save_directory,exist_ok=True)
#     create_table(final_dataframes_by_run[i],10,save_directory)
#     create_table(final_dataframes_by_run[i],50,save_directory)
#     create_table(final_dataframes_by_run[i],100,save_directory)
#     create_table(final_dataframes_by_run[i],1000,save_directory)
#     zs = avg_df.copy()
#     zs['value']=(final_dataframes_by_run[i]['value']-avg_df['value'])#/sd['value']
#     zs['value_optimal']=(final_dataframes_by_run[i]['value_optimal']-avg_df['value_optimal'])#/sd['value_optimal']
#     print(i)
#     for dataset, model in unique_combinations:
#         plot_lines(zs,dataset,model,df['k_fp'].max(),save_directory)

'''
Copyright 2024 Nicolò Marchini
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''