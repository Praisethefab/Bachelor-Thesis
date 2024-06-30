import os
import re
from PIL import Image, ImageDraw, ImageFont

for i in ['plot_lines','platt_lines','bbq_lines','base_lines','his_lines','iso_lines']:
    folder_path = os.path.join('.', 'value', i)

    images_dict = {}
    pattern = r"^(.*)_([^_]+)_plot.png$"
    databases = []
    models = []
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            database, model = match.groups()
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            if model not in models:
                models.append(model)

            if database not in images_dict:
                images_dict[database] = []
            images_dict[database].append(img)

    # Reorganize images_dict to have databases as keys
    databases = list(images_dict.keys())

    # Calculate grid size and max columns
    grid_size = len(databases)
    max_columns = max(len(images) for images in images_dict.values())

    # Create a new grid image
    grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

    # Paste images onto the grid
    for i, database in enumerate(databases):
        for j, img in enumerate(images_dict[database]):
            grid_image.paste(img, (j * img.width, i * img.height))

    # Add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype('arial.ttf', 60)

    # Add labels for rows (databases)
    for i, database in enumerate(databases):
        draw.text((0, i * img.height), database, font=font, fill=(0, 0, 0))

    # Add labels for columns (models)
    for j in range(max_columns):
        draw.text((j * img.width + img.width/2.3, 0), models[j], font=font, fill=(0, 0, 0))

    grid_image_path = os.path.join(folder_path, 'Plot_grid.png')
    grid_image.save(grid_image_path)

    print(f"Grid image with labels saved at {grid_image_path}")

for i in [ "10.0_100", "100_500", "500_1000.0"]:
    folder_path = os.path.join('.', 'value', "plot_lines", i)

    images_dict = {}
    pattern = r"^(.*)_([^_]+)_plot.png$"
    databases = []
    models = []
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            database, model = match.groups()
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            if model not in models:
                models.append(model)

            if database not in images_dict:
                images_dict[database] = []
            images_dict[database].append(img)

    # Reorganize images_dict to have databases as keys
    databases = list(images_dict.keys())

    # Calculate grid size and max columns
    grid_size = len(databases)
    max_columns = max(len(images) for images in images_dict.values())

    # Create a new grid image
    grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

    # Paste images onto the grid
    for i, database in enumerate(databases):
        for j, img in enumerate(images_dict[database]):
            grid_image.paste(img, (j * img.width, i * img.height))

    # Add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype('arial.ttf', 60)

    # Add labels for rows (databases)
    for i, database in enumerate(databases):
        draw.text((0, i * img.height), database, font=font, fill=(0, 0, 0))

    # Add labels for columns (models)
    for j in range(max_columns):
        draw.text((j * img.width + img.width/2.3, 0), models[j], font=font, fill=(0, 0, 0))

    grid_image_path = os.path.join(folder_path, 'Plot_grid.png')
    grid_image.save(grid_image_path)

    print(f"Grid image with labels saved at {grid_image_path}")

for z in ["42", "938465", "652894", "134875", "394856", "596657", "657492", "938563", "678430", "578231"]:
    folder_path = os.path.join('.', 'value', z, "plot")

    images_dict = {}
    pattern = r"^(.*)_([^_]+)_plot.png$"
    databases = []
    models = []
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            database, model = match.groups()
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            if model not in models:
                models.append(model)

            if database not in images_dict:
                images_dict[database] = []
            images_dict[database].append(img)

    # Reorganize images_dict to have databases as keys
    databases = list(images_dict.keys())

    # Calculate grid size and max columns
    grid_size = len(databases)
    max_columns = max(len(images) for images in images_dict.values())

    # Create a new grid image
    grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

    # Paste images onto the grid
    for i, database in enumerate(databases):
        for j, img in enumerate(images_dict[database]):
            grid_image.paste(img, (j * img.width, i * img.height))

    # Add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype('arial.ttf', 60)

    # Add labels for rows (databases)
    for i, database in enumerate(databases):
        draw.text((0, i * img.height), database, font=font, fill=(0, 0, 0))

    # Add labels for columns (models)
    for j in range(max_columns):
        draw.text((j * img.width + img.width/2.3, 0), models[j], font=font, fill=(0, 0, 0))

    grid_image_path = os.path.join(folder_path, 'Plot_grid.png')
    grid_image.save(grid_image_path)

    print(f"Grid image with labels saved at {grid_image_path}")
    
    for j in [ "10.0_100", "100_500", "500_1000.0"]:
        folder_path = os.path.join('.', 'value', z, "plot", j)

        images_dict = {}
        pattern = r"^(.*)_([^_]+)_plot.png$"
        databases = []
        models = []
        for filename in os.listdir(folder_path):
            match = re.match(pattern, filename)
            if match:
                database, model = match.groups()
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                if model not in models:
                    models.append(model)

                if database not in images_dict:
                    images_dict[database] = []
                images_dict[database].append(img)

        # Reorganize images_dict to have databases as keys
        databases = list(images_dict.keys())

        # Calculate grid size and max columns
        grid_size = len(databases)
        max_columns = max(len(images) for images in images_dict.values())

        # Create a new grid image
        grid_image = Image.new('RGB', (max_columns * img.width, grid_size * img.height))

        # Paste images onto the grid
        for i, database in enumerate(databases):
            for j, img in enumerate(images_dict[database]):
                grid_image.paste(img, (j * img.width, i * img.height))

        # Add labels to the grid
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.truetype('arial.ttf', 60)

        # Add labels for rows (databases)
        for i, database in enumerate(databases):
            draw.text((0, i * img.height), database, font=font, fill=(0, 0, 0))

        # Add labels for columns (models)
        for j in range(max_columns):
            draw.text((j * img.width + img.width/2.3, 0), models[j], font=font, fill=(0, 0, 0))

        grid_image_path = os.path.join(folder_path, 'Plot_grid.png')
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