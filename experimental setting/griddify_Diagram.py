import os
import re
from PIL import Image, ImageDraw, ImageFont

# Define the regex pattern to match the folder names
pattern = re.compile(r'graphs_(\d+)')

# Regular expression to match model and calibrator
pattern2 = re.compile(r'^(.+?)_(.+?)\.png$')

# Iterate over all items in the base path
for folder_path in os.listdir(os.getcwd()):
    if os.path.isdir(folder_path) and pattern.match(folder_path):
        grid_dir = os.path.join(os.getcwd(), "grids", f'{folder_path}_grid')
        os.makedirs(grid_dir, exist_ok=True)

        # Dictionary to hold images for each dataset
        images_dict = {}

        for dataset in os.listdir(folder_path):
            if (dataset=="table"):
                continue
            dataset_path = os.path.join(folder_path, dataset, "ReliabilityDiagram")
            if dataset not in images_dict:
                images_dict[dataset] = {}
            
            for filename in os.listdir(dataset_path):
                match = pattern2.match(filename)
                if match:
                    model, calibrator = match.groups()
                    if calibrator == "Curve":
                        continue # Skip curve images
                    img_path = os.path.join(dataset_path, filename)
                    img = Image.open(img_path)
                    
                    if model not in images_dict[dataset]:
                        images_dict[dataset][model] = []
                    images_dict[dataset][model].append((calibrator, img))
            
            # Add base images for each model in the dataset
            for model in images_dict[dataset].keys():
                base_img_path = os.path.join(dataset_path, f"{model}.png")
                if os.path.exists(base_img_path):
                    base_img = Image.open(base_img_path)
                    images_dict[dataset][model].append(("Base", base_img))

            # Sort images for each model to ensure "Base" is first
            for model in images_dict[dataset].keys():
                images_dict[dataset][model].sort(key=lambda x: (x[0] != "Base", x[0]))
        # Create separate images for each dataset
        for dataset, models in images_dict.items():
            # Determine the number of rows and columns
            num_rows = len(models)
            num_columns = max(len(images) for images in models.values())
            
            # Create a new grid image
            grid_image = Image.new('RGB', (num_columns * models[next(iter(models))][0][1].width, 
                                        10+num_rows * models[next(iter(models))][0][1].height), color="white")
            unique_calibrators = set()
            # Paste images into the grid
            for i, (model, images) in enumerate(models.items()):
                for j, (calibrator, img) in enumerate(images):
                    if calibrator not in unique_calibrators:
                        unique_calibrators.add(calibrator)
                    grid_image.paste(img, (j * img.width, 10+i * img.height))
            
            # Add labels to the grid
            draw = ImageDraw.Draw(grid_image)
            font = ImageFont.truetype('arial.ttf', 20) # Adjust the font and size as needed
            
            # Add labels for rows (models)
            for i, model in enumerate(models.keys()):
                draw.text((0, i * models[model][0][1].height), model, font=font, fill=(0, 0, 0))
        
            # Now, iterate over the unique calibrator names to draw the labels
            for j, calibrator in enumerate(unique_calibrators):
                draw.text((j * models[next(iter(models))][0][1].width + models[next(iter(models))][0][1].width/2, 0), calibrator, font=font, fill=(0, 0, 0))
            
            # Save the grid image
            grid_image_path = os.path.join(grid_dir, f'{dataset}_grid.png')
            grid_image.save(grid_image_path)
            print(f"Grid image for dataset {dataset} saved at {grid_image_path}\n")

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