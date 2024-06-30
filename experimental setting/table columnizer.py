import os
import re
from PIL import Image

base_dir=os.getcwd()

# Initialize a dictionary to hold images for each unique identifier
images_dict = {}

dir_pattern = re.compile(r"graphs_(\d+)")

 # Iterate over all items in the base directory
for item in os.listdir(base_dir):
    # Check if the item is a directory and matches the pattern
    if os.path.isdir(os.path.join(base_dir, item)) and dir_pattern.match(item):
        # Construct the full path to the directory
        dir_path = os.path.join(base_dir, item, "table")
        
        # Iterate over all items in the directory
        for filename in os.listdir(dir_path):
            # Check if the item matches the pattern
            match = re.match(r"Table_(\w+).png", filename)
            if match:
                # Extract the unique identifier
                identifier = match.group(1)
                
                # Open the image and add it to the dictionary
                if identifier not in images_dict:
                    images_dict[identifier] = []
                images_dict[identifier].append(Image.open(os.path.join(dir_path, filename)))

# Process each unique identifier
for identifier, images in images_dict.items():
    # Calculate the size of the combined image based on the largest image
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # Create a new blank image with the calculated size
    combined_image = Image.new('RGB', (max_width, total_height))
    
    # Paste each image onto the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    save_dir=os.path.join(base_dir,"columns")
    os.makedirs(save_dir,exist_ok=True)
    # Save the combined image
    combined_image.save(os.path.join(save_dir,f'combined_images_{identifier}.png'))

base_dir=os.path.join(os.getcwd(),"value")

# Initialize a dictionary to hold images for each unique identifier
images_dict = {}

dir_pattern = re.compile(r"(\d+)")

 # Iterate over all items in the base directory
for item in os.listdir(base_dir):
    # Check if the item is a directory and matches the pattern
    if os.path.isdir(os.path.join(base_dir, item)) and dir_pattern.match(item):
        # Construct the full path to the directory
        dir_path = os.path.join(base_dir, item, "auv_tables")
        
        # Iterate over all items in the directory
        for filename in os.listdir(dir_path):
            # Check if the item matches the pattern
            match = re.match(r"AUV_Table_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?).png", filename)
            if match:
                # Extract the unique identifier
                identifier = f"{match.group(1)}_{match.group(2)}"
                
                # Open the image and add it to the dictionary
                if identifier not in images_dict:
                    images_dict[identifier] = []
                images_dict[identifier].append(Image.open(os.path.join(dir_path, filename)))

# Process each unique identifier
for identifier, images in images_dict.items():
    # Calculate the size of the combined image based on the largest image
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # Create a new blank image with the calculated size
    combined_image = Image.new('RGB', (max_width, total_height))
    
    # Paste each image onto the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    save_dir=os.path.join(base_dir,"columns")
    os.makedirs(save_dir,exist_ok=True)
    # Save the combined image
    combined_image.save(os.path.join(save_dir,f'combined_images_{identifier}.png'))

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