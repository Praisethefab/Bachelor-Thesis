import os
import re
from PIL import Image

# Define the regex pattern to match the folder names
pattern = re.compile(r'graphs_(\d+)')

# Regular expression to match model and calibrator
pattern2 = re.compile(r'^(.+?)_(.+?)\.png$')

# Iterate over all items in the base path
base_path = os.getcwd()
output_base_path = os.path.join(base_path, "grids")

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Dictionary to hold images for each dataset
images_dict = {}

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    if os.path.isdir(folder_path) and pattern.match(folder_name):
        random_state = pattern.match(folder_name).group(1)
        
        for dataset in os.listdir(folder_path):
            if dataset == "table":
                continue
            
            dataset_path = os.path.join(folder_path, dataset, "ReliabilityDiagram")
            if dataset not in images_dict:
                images_dict[dataset] = {}
            
            for filename in os.listdir(dataset_path):
                match = pattern2.match(filename)
                if match:
                    model, calibrator = match.groups()
                    if calibrator == "Curve":
                        continue  # Skip curve images
                    img_path = os.path.join(dataset_path, filename)
                    img = Image.open(img_path)
                    
                    if model not in images_dict[dataset]:
                        images_dict[dataset][model] = {}
                    if random_state not in images_dict[dataset][model]:
                        images_dict[dataset][model][random_state] = {}
                    images_dict[dataset][model][random_state][calibrator] = img
            
            # Add base images for each model in the dataset
            for model in images_dict[dataset].keys():
                base_img_path = os.path.join(dataset_path, f"{model}.png")
                if os.path.exists(base_img_path):
                    base_img = Image.open(base_img_path)
                    if random_state not in images_dict[dataset][model]:
                        images_dict[dataset][model][random_state] = {}
                    images_dict[dataset][model][random_state]["Base"] = base_img

# Create grids for each dataset and model
for dataset, models in images_dict.items():
    output_dataset_path = os.path.join(output_base_path, dataset)
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)
    
    for model, states in models.items():
        random_states = sorted(states.keys())
        calibrators = sorted(list({cal for state_imgs in states.values() for cal in state_imgs}))

        max_calibrators = len(calibrators)
        max_random_states = len(random_states)

        if max_calibrators == 0 or max_random_states == 0:
            continue

        # Determine the size of the grid
        first_image = states[random_states[0]][calibrators[0]]
        img_width, img_height = first_image.size
        grid_width = img_width * max_calibrators
        grid_height = img_height * max_random_states

        # Create the grid image
        #print(f"{grid_width} , {grid_height} \n")
        grid_img = Image.new('RGB', (grid_width, grid_height))
        for r_idx, r_state in enumerate(random_states):
            for c_idx, calibrator in enumerate(calibrators):
                if calibrator in states[r_state]:
                    img = states[r_state][calibrator]
                    grid_img.paste(img, (c_idx * img_width, r_idx * img_height))

        output_image_path = os.path.join(output_dataset_path, f"{model}.png")
        grid_img.save(output_image_path)
        grid_img.close()
        del grid_img
        print(f"Grid image for dataset {model} saved at {output_image_path}\n")

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