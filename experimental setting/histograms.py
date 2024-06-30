import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math

# Choose the DPI of the graphs you want to print
DPI = 300

# Directory where the CSV files are located
csv_dir = os.getcwd()
# Regular expression to match the CSV files
csv_pattern = re.compile(r'scores_(\d+)\.csv')

# Find all matching CSV files
csv_files = [f for f in os.listdir(csv_dir) if csv_pattern.match(f)]

for csv_file in csv_files:
    # Extract the random_state from the filename
    random_state = csv_pattern.search(csv_file).group(1)
    print({random_state})

    # Read the CSV file
    df = pd.read_csv(os.path.join(csv_dir, csv_file))
    
    # Iterate over each combination of classificator and dataset_tag
    for dataset_tag in df['dataset_tag'].unique():
        output_dir = os.path.join(csv_dir, f"graphs_{random_state}", dataset_tag, "histograms")
        for classificator in df['classificator'].unique():
            # Filter DataFrame for specific classificator and dataset_tag
            filtered_df = df[(df['classificator'] == classificator) & (df['dataset_tag'] == dataset_tag)]
            
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Set bar width
            bar_width = 0.30
            
            # Positions of the left bar-boundaries for the first category
            tick_pos = [i for i, _ in enumerate(filtered_df['calibrator'])]
            
            # Create a color map
            colors = {'TP': 'tab:green', 'FN': 'tab:cyan', 'FP': 'tab:orange'}
            
            # Plot each category
            for idx, category in enumerate(['TP', 'FN', 'FP']):
                # Adjust tick_pos for each category
                adjusted_tick_pos = [pos + idx * bar_width for pos in tick_pos]
                
                ax.bar(adjusted_tick_pos, filtered_df[category], color=colors[category], width=bar_width, label=category)

                # Iterate through each bar and its corresponding value
                for i, value in enumerate(filtered_df[category]):
                    # Check if the value is not zero
                    if value!= 0:
                        # Calculate the position for the label
                        x_pos = adjusted_tick_pos[i]
                        y_pos = value / 2  # Adjust this to position the label inside the bar
                        # Add the label
                        ax.text(x_pos, y_pos, str(value), ha='center', va='center', color='black')
            
            # Set x-axis labels
            plt.xticks([pos + bar_width / 2 for pos in tick_pos], filtered_df['calibrator'])
            
            # Set title and labels
            last_row = filtered_df.iloc[-1]
            plt.title(f"{random_state} - {dataset_tag} - {classificator} total = {last_row["TN"]+last_row["TP"]+last_row["FN"]+last_row["FP"]}")
            plt.xlabel('Calibrators')
            plt.ylabel('Quantity')
            plt.legend()

            os.makedirs(output_dir, exist_ok=True)
            filename = f"{classificator}_{dataset_tag}.png"
            plt.savefig(os.path.join(output_dir, filename), format="png", dpi=DPI)
            plt.close(fig)

def create_grid_image(model, model_dict):
    max_cols = max([len(random_dict) for random_dict in model_dict.values()])
    max_rows = len(model_dict)
    
    width = math.ceil(800*DPI/100)  # Width of each cell in pixels
    height = math.ceil(600*DPI/100)  # Height of each cell in pixels
    
    total_width = width * max_cols
    total_height = height * max_rows
    
    new_img = Image.new('RGB', (total_width, total_height))
    
    x_offset = 0
    y_offset = 0
    
    for random_state, random_dict in model_dict.items():
        for database, image_path in random_dict.items():
            img = Image.open(image_path)
            new_img.paste(img, (x_offset, y_offset))
            
            x_offset += width
            
            if x_offset >= total_width:
                x_offset = 0
                y_offset += height
    save_dir=os.path.join(os.getcwd(), "grids")
    os.makedirs(save_dir,exist_ok=True)
    new_img.save(os.path.join(os.getcwd(), "grids", f'{model}_histograms.png'))

base_dir = os.getcwd()
# Nested dictionary to hold images grouped by model, database, and random_state
image_dict = {}

# Pattern to extract random state from directory name
pattern = re.compile(r'graphs_(\d+)')

# Traverse directories
for dir in os.listdir(base_dir):
    if pattern.match(dir):
        random_state = pattern.match(dir).group(1)  # Extract random state
        dir_path = os.path.join(base_dir, dir)
        for dir2 in os.listdir(dir_path):
            if dir2!= "table":
                histograms_path = os.path.join(dir_path, dir2, "histograms")
                for file in os.listdir(histograms_path):
                    if file.endswith('.png'):
                        model_match = re.search(r'[a-zA-Z]+', file)
                        database_match = re.search(r'_[a-zA-Z0-9_ ]+', file)
                        if model_match and database_match:
                            model = model_match.group()
                            database = database_match.group().lstrip('_')
                            # Organize images into the nested dictionary
                            if (model) not in image_dict:
                                image_dict[model] = {}
                            if (random_state) not in image_dict[model]:
                                image_dict[model][random_state] = {}
                            image_path = os.path.join(histograms_path, file)
                            image_dict[model][random_state].update({database: image_path})
for model, model_dict in image_dict.items():
    create_grid_image(model, model_dict)
    print(f"{model}")


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
