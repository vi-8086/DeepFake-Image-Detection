import shutil
import os

# Get the path to the extracted 'Dataset' directory
extracted_dataset_directory = os.path.join(extract_to_directory, 'Dataset')

# List the contents of the 'Dataset' directory
dataset_contents = os.listdir(extracted_dataset_directory)

# Move each item from the 'Dataset' directory to the extraction directory
for item in dataset_contents:
    # Get the full path of the item
    item_path = os.path.join(extracted_dataset_directory, item)
    
    # Move the item to the extraction directory
    shutil.move(item_path, extract_to_directory)

# Remove the empty 'Dataset' directory
os.rmdir(extracted_dataset_directory)

# List the extracted files again
extracted_files = os.listdir(extract_to_directory)
print("Extracted files:", extracted_files)

