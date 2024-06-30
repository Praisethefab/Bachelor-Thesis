import os
import re
from PIL import Image
import numpy as np

# Define the regex pattern to match the folder names
pattern = re.compile(r'graphs_(\d+)')

# Create the output directory if it doesn't exist
output_dir = os.path.join(os.getcwd(), "grids", "rocs")
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to hold images for each model
images_dict = {}

# Iterate over all items in the base path
for folder_name in os.listdir(os.getcwd()):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if os.path.isdir(folder_path) and pattern.match(folder_name):
        random_state = pattern.match(folder_name).group(1)

        for dataset in os.listdir(folder_path):
            if dataset == "table":
                continue
            dataset_path = os.path.join(folder_path, dataset, "roc")
            
            for filename in os.listdir(dataset_path):
                match = re.match(r'^(.*)_([^_]+)_roc.png$', filename)
                if match:
                    dataset_name, model = match.groups()
                    img_path = os.path.join(dataset_path, filename)
                    img = Image.open(img_path)

                    if model not in images_dict:
                        images_dict[model] = {}
                    if dataset_name not in images_dict[model]:
                        images_dict[model][dataset_name] = {}
                    images_dict[model][dataset_name][random_state] = img

# Create and save grids for each model
for model, datasets in images_dict.items():
    # Determine the grid size
    num_datasets = len(datasets)
    num_random_states = max(len(random_states) for random_states in datasets.values())

    # Determine the dimensions of the images (assuming all images are the same size)
    first_img = next(iter(next(iter(datasets.values())).values()))
    img_width, img_height = first_img.size

    # Create an empty canvas for the grid
    grid_width = num_datasets * img_width
    grid_height = num_random_states * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Populate the grid with images
    for col, (dataset_name, random_states) in enumerate(datasets.items()):
        for row, random_state in enumerate(sorted(random_states.keys(), key=int)):
            img = random_states[random_state]
            grid_image.paste(img, (col * img_width, row * img_height))

    # Save the grid image
    output_path = os.path.join(output_dir, f"{model}_grid.png")
    grid_image.save(output_path)

print(f"Grids created and saved in {output_dir}")

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