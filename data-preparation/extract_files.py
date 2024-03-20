import zipfile
import os
import random
import shutil

def count_files(directory):
    counts = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            subdir = os.path.basename(root)
            class_name = os.path.basename(os.path.dirname(root))
            counts.setdefault(class_name, {})
            counts[class_name].setdefault(subdir, 0)
            counts[class_name][subdir] += 1
    return counts

def extract_files(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    before_extraction = count_files(extract_dir)

    extracted_files_subset = []
    for class_name, subdir_counts in before_extraction.items():
        for subdir, count in subdir_counts.items():
            subdir_path = os.path.join(extract_dir, 'Dataset', class_name, subdir)
            if os.path.exists(subdir_path):  # Check if subdirectory exists
                files_to_extract = random.sample(os.listdir(subdir_path), int(0.1 * count))
                for file in files_to_extract:
                    source_file = os.path.join(subdir_path, file)
                    destination_dir = os.path.join(extract_dir, 'Subset', 'Dataset', class_name, subdir)
                    os.makedirs(destination_dir, exist_ok=True)
                    destination_file = os.path.join(destination_dir, file)
                    shutil.copy(source_file, destination_file)
                    extracted_files_subset.append(destination_file)

    after_extraction = count_files(os.path.join(extract_dir, 'Subset'))

    return before_extraction, after_extraction

if __name__ == "__main__":
    zip_file = "/content/drive/MyDrive/Colab Notebooks/DeepFake Image Detection/DFDC.zip"  
    extract_dir = "/content/extracted_files" 

    before_extraction, after_extraction = extract_files(zip_file, extract_dir)

    print("Initial count of files in each class before extraction:")
    for class_name, subdir_counts in before_extraction.items():
        print(f"{class_name}:")
        for subdir, count in subdir_counts.items():
            print(f"  {subdir}: {count}")

    print("\nCount of files after extracting 10% while preserving balance:")
    for class_name, subdir_counts in after_extraction.items():
        print(f"{class_name}:")
        for subdir, count in subdir_counts.items():
            print(f"  {subdir}: {count}")
