import os
import re
from PIL import Image, ImageDraw, ImageFont

# Define the regex pattern to match the folder names
pattern = re.compile(r'graphs_(\d+)')

# Iterate over all items in the base path
for folder_path in os.listdir(os.getcwd()):
    if os.path.isdir(folder_path) and pattern.match(folder_path):
        grid_dir = os.path.join(os.getcwd(), "grids", f'{folder_path}_grid')
        os.makedirs(grid_dir, exist_ok=True)

        images_dict = {}

        for dataset in os.listdir(folder_path):
            if (dataset=="table"):
                continue
            dataset_path = os.path.join(folder_path, dataset, "ReliabilityDiagram")
            for filename in os.listdir(dataset_path):
                if re.match(r'^.*_Curve\.png$', filename):
                    model = filename.split('_')[0]
                    img_path = os.path.join(dataset_path, filename)
                    img = Image.open(img_path)
                    
                    if model not in images_dict:
                        images_dict[model] = []
                    images_dict[model].append(img)
                    

        grid_size = len(images_dict)
        max_columns = max(len(images) for images in images_dict.values())

        grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

        for i, (model, images) in enumerate(images_dict.items()):
            for j, img in enumerate(images):
                grid_image.paste(img, (j * img.width, i * img.height))

        # Add labels to the grid
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.truetype('arial.ttf', 20)

        # Add labels for rows (models)
        for i, model in enumerate(images_dict.keys()):
            draw.text((0, i * img.height), model, font=font, fill=(0, 0, 0))

        # Add labels for columns (folder paths)
        for j in range(max_columns):
            draw.text((j * img.width + img.width/2.5, 0), os.listdir(folder_path)[j], font=font, fill=(0, 0, 0))

        grid_image_path = os.path.join(grid_dir, f'Curve_grid.png')
        grid_image.save(grid_image_path)

        print(f"Grid image with labels saved at {grid_image_path}")


        images_dict = {}

        for dataset in os.listdir(folder_path):
            if (dataset=="table"):
                continue
            dataset_path = os.path.join(folder_path, dataset, "roc")
            for filename in os.listdir(dataset_path):
                match=re.match(r'^(.*)_([^_]+)_roc.png$', filename)
                if match:
                    _, model = match.groups()
                    img_path = os.path.join(dataset_path, filename)
                    img = Image.open(img_path)
                    if model not in images_dict:
                        images_dict[model] = []
                    images_dict[model].append(img)
                    

        grid_size = len(images_dict)
        max_columns = max(len(images) for images in images_dict.values())

        grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

        for i, (model, images) in enumerate(images_dict.items()):
            for j, img in enumerate(images):
                grid_image.paste(img, (j * img.width, i * img.height))

        # Add labels to the grid
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.truetype('arial.ttf', 20)

        # Add labels for rows (models)
        for i, model in enumerate(images_dict.keys()):
            draw.text((0, i * img.height), model, font=font, fill=(0, 0, 0))

        # Add labels for columns (folder paths)
        for j in range(max_columns):
            draw.text((j * img.width + img.width/2.5, 0), os.listdir(folder_path)[j], font=font, fill=(0, 0, 0))

        grid_image_path = os.path.join(grid_dir, f'roc_grid.png')
        grid_image.save(grid_image_path)

        print(f"Grid image with labels saved at {grid_image_path}")

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