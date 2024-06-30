import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import dataframe_image as dfi
import os
import re

# Choose the DPI of the graphs you want to print
DPI = 300


def create_table(column_name, file_names, dir, savedir):
    # Initialize a DataFrame to store the averages
    avg_df = pd.read_csv(os.path.join(dir,file_names[0]))
    avg_df = avg_df[["dataset_tag", "classificator", "calibrator", column_name]]
    avg_df[column_name] = pd.to_numeric(avg_df[column_name], errors='coerce')
    if(len(file_names)>1):
        name="A verage"
    else:
        name=f"Random State {re.match(r"scores_(\d+).csv", file_names[0]).group(1)}"

    # Loop through each file name
    for file_name in file_names[1:]:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(dir,file_name))
        # Select the required columns
        df = df[["dataset_tag", "classificator", "calibrator", column_name]]
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        

        # Calculate the average of the column across all files
        avg_df[column_name] += df[column_name].fillna(0)


    # Calculate the average across all files
    avg_df[column_name] = avg_df[column_name] / len(file_names)
    
    # Create a pivot table from the averaged DataFrame
    pivot_df = avg_df.pivot_table(index='classificator', columns=['dataset_tag', 'calibrator'], values=column_name)
    # Get the unique values of 'calibrator' and 'dataset_tag'
    calibrator_values = pivot_df.columns.get_level_values('calibrator').unique()
    dataset_tags = pivot_df.columns.get_level_values('dataset_tag').unique()

    # Check if 'Base' is in the list of unique 'calibrator' values
    if 'Base' in calibrator_values:
        # Create a new list of tuples for the new multi-level index
        new_columns = []
        for tag in dataset_tags:
            # Add 'Base' first for each 'dataset_tag'
            new_columns.append((tag, 'Base'))
            # Add the rest of the 'calibrator' values for each 'dataset_tag'
            for calibrator in calibrator_values:
                if calibrator != 'Base':
                    new_columns.append((tag, calibrator))
        
    # Create a new multi-level index
    new_index = pd.MultiIndex.from_tuples(new_columns, names=['dataset_tag', 'calibrator'])
    
    # Reorder the columns of the DataFrame using the new multi-level index
    pivot_df = pivot_df.reindex(columns=new_index)
    # Initialize an empty DataFrame with the same structure as pivot_df to store the normalized values
    normalized_pivot_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

    # # Determine the min and max values for the gradient
    # M=pivot_df.max().max()
    # m=pivot_df.min().min()
    # if column_name=="acc":
    #     m=0.90
    # elif column_name=="ece":
    #     M=0.085
    # elif column_name=="agf":
    #     m=0.57
    # elif column_name=="f1" or column_name=="gmean":
    #     m=0.25
    # elif column_name=="gini":
    #     m=0.25
    # elif column_name=="mcc":
    #     m=0.30

    # Iterate over each index level in the pivot table
    for model in pivot_df.index.unique():
        # Filter the pivot_df for the current model
        model_df = pivot_df.loc[model]
        # Normalize the values for each column in the current model
        for column in pivot_df.columns.get_level_values("dataset_tag").unique():
            normalized_values = (model_df[column] - model_df[column].min().min()) / (model_df[column].max().max() - model_df[column].min().min())
            # Assign the normalized values to the corresponding column in the normalized DataFrame 
            normalized_pivot_df.loc[model, column] = normalized_values.to_numpy()

    # # Iterate over each column in the pivot table
    # for column in pivot_df.columns.get_level_values('dataset_tag').unique():
    #     #filters the data to exclude the values that are anomalous
    #     filtered_values = pivot_df[column][(pivot_df[column] >= m) & (pivot_df[column] <= M)]
    #     min_value = filtered_values.min().min().min()
    #     max_value = filtered_values.max().max().max()
    #     # Normalize the values in the current column
    #     normalized_values = (pivot_df[column] - min_value) / (max_value - min_value)
    #     normalized_values = np.where((normalized_values < 0) | (normalized_values > 1), 0, normalized_values)

    #     # Assign the normalized values to the corresponding column in the normalized DataFrame
    #     normalized_pivot_df[column] = normalized_values

    pd.set_option('future.no_silent_downcasting', True)
    normalized_pivot_df=normalized_pivot_df.fillna(1)

    # apply gradient
    if column_name=="ece":
        cmap = LinearSegmentedColormap.from_list("green_to_red", ["green","white", "#FF3333"],N=65536,)
        styled_df=pivot_df.style.background_gradient(cmap=cmap,axis=None, gmap=normalized_pivot_df)
    else:
        cmap = LinearSegmentedColormap.from_list("red_to_green", ["#FF3333","white", "green"],N=65536,)
        styled_df=pivot_df.style.background_gradient(cmap=cmap,axis=None, gmap=normalized_pivot_df)
    
    #print(pivot_df.loc["GaussianNB"])

    # Iterate over each 'dataset_tag'
    for dataset_tag in pivot_df.columns.get_level_values(0).unique():
        if column_name=="ece":
            styled_df.highlight_min(subset=dataset_tag, props='color: black;  font-weight: bold;', axis=1)
        else:
            styled_df.highlight_max(subset=dataset_tag, props='color: black;  font-weight: bold;', axis=1)

    styled_df = styled_df.set_properties(**{'color': 'black'})

    # def anomaliesm(val):
    #     if val<m:
    #         return 'background-color: #E4D96F; color: black;'
    #     else:
    #         return ''
    # def anomaliesM(val):
    #     if val>M:
    #         return 'background-color: #E4D96F; color: black;'
    #     else:
    #         return ''
        
    # if column_name=="ece":
    #     styled_df = styled_df.map(anomaliesM)
    # else:
    #     styled_df = styled_df.map(anomaliesm)

    match column_name:
        case 'acc':
            extended_name = "Accuracy"
        case 'mcc':
            extended_name = "Matthews Correlation Coefficient"
        case 'ece':
            extended_name = "Expected Calibration Error"
        case 'agf':
            extended_name = "Adjusted F-Score"
        case 'f1':
            extended_name = "F1 Score"
        case 'gmean':
            extended_name = "Geometric Mean"
        case 'gini':
            extended_name = "Gini Coefficent"
        case _:
            extended_name = "Unknown Metric"

    title = f"{name} {extended_name} Table, otherwise it goes from red (bad) to green (good) with the bold ones being the best on that model and dataset"
    styled_df = styled_df.set_caption(title)
    styled_df = styled_df.set_table_styles([
        {'selector': 'td', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(6)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(11)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(16)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(21)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(6)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(11)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(16)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(21)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'caption', 'props': [('font-size', '20px bold'),
                                          ('font-weight', 'bold'), ('padding', '5px')]},
    ])

    # Apply the formatter to the DataFrame using the Styler object
    styled_df = styled_df.format(precision=5)
    os.makedirs(os.path.join(dir,f"table"), exist_ok=True)
    dfi.export(styled_df, os.path.join(savedir,f'Table_{column_name}.png'), dpi=DPI)
    
    if(len(file_names)<2):
        return

    # Initialize DataFrames to store the sums and sums of squares
    sum_df = pd.read_csv(os.path.join(dir, file_names[0]))
    sum_df = sum_df[["dataset_tag", "classificator", "calibrator", column_name]]
    sum_df[column_name] = pd.to_numeric(sum_df[column_name], errors='coerce').fillna(0)
    sum_of_squares_df = sum_df.copy()
    sum_of_squares_df[column_name] = sum_df[column_name] ** 2

    # Loop through each file name to update sums and sums of squares
    for file_name in file_names[1:]:
        df = pd.read_csv(os.path.join(dir, file_name))
        df = df[["dataset_tag", "classificator", "calibrator", column_name]]
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
        
        sum_df[column_name] += df[column_name]
        sum_of_squares_df[column_name] += df[column_name] ** 2

    # Calculate the variance
    N = len(file_names)
    mean_squared = sum_df[column_name] ** 2 / N
    variance = (sum_of_squares_df[column_name] - mean_squared) / N
    # Create a DataFrame to store the variance
    var_df = sum_df.copy()
    var_df[column_name] = variance

    # Create a pivot table from the variance DataFrame
    pivot_var_df = var_df.pivot_table(index='classificator', columns=['dataset_tag', 'calibrator'], values=column_name)

    # Get the unique values of 'calibrator' and 'dataset_tag'
    calibrator_values = pivot_var_df.columns.get_level_values('calibrator').unique()
    dataset_tags = pivot_var_df.columns.get_level_values('dataset_tag').unique()

    # Check if 'Base' is in the list of unique 'calibrator' values
    if 'Base' in calibrator_values:
        # Create a new list of tuples for the new multi-level index
        new_columns = []
        for tag in dataset_tags:
            # Add 'Base' first for each 'dataset_tag'
            new_columns.append((tag, 'Base'))
            # Add the rest of the 'calibrator' values for each 'dataset_tag'
            for calibrator in calibrator_values:
                if calibrator != 'Base':
                    new_columns.append((tag, calibrator))

    # Create a new multi-level index
    new_index = pd.MultiIndex.from_tuples(new_columns, names=['dataset_tag', 'calibrator'])
    
    # Reorder the columns of the DataFrame using the new multi-level index
    pivot_var_df = pivot_var_df.reindex(columns=new_index)
    
    styled_df = pivot_var_df.style.set_properties(**{'color': 'black'})

    title = f"{name} {extended_name} variance Table"
    styled_df = styled_df.set_caption(title)
    styled_df = styled_df.set_table_styles([
        {'selector': 'td', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th', 'props': [('border', '1px solid black'),('padding', '2px')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(6)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(11)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(16)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'th:nth-child(21)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(6)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(11)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(16)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'td:nth-child(21)', 'props': [('border-right', '2px solid black')]},
        {'selector': 'caption', 'props': [('font-size', '20px bold'),
                                          ('font-weight', 'bold'), ('padding', '5px')]},
    ])

    # Apply the formatter to the DataFrame using the Styler object
    styled_df = styled_df.format(precision=5)
    dfi.export(styled_df, os.path.join(savedir,f'Table_variance_{column_name}.png'), dpi=DPI)

list_of_files=[]
dir=os.path.join(os.getcwd())
os.makedirs(dir, exist_ok=True)
for filename in os.listdir(dir):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Match the filename against the pattern
        match = re.match(r"scores_(\d+).csv", filename)
        if(match):
            print(filename)
            list_of_files.append(filename)
            savedir=os.path.join(os.getcwd(),f"graphs_{match.group(1)}","table")
            os.makedirs(savedir, exist_ok=True)
            create_table('acc',[filename],dir,savedir)
            create_table('mcc',[filename],dir,savedir)
            create_table('ece',[filename],dir,savedir)
            create_table('agf',[filename],dir,savedir)
            create_table('f1',[filename],dir,savedir)
            # create_table('gmean',[filename],dir,savedir)
            create_table('gini',[filename],dir,savedir)

savedir=os.path.join(os.getcwd(),"table")
os.makedirs(savedir, exist_ok=True)

create_table('acc',list_of_files,dir,savedir)
create_table('mcc',list_of_files,dir,savedir)
create_table('ece',list_of_files,dir,savedir)
create_table('agf',list_of_files,dir,savedir)
create_table('f1',list_of_files,dir,savedir)
# create_table('gmean',list_of_files,dir,savedir)
create_table('gini',list_of_files,dir,savedir)

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