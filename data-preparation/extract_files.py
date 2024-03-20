import zipfile
import os
import random

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

    extracted_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    num_files_to_extract = {}
    for class_name, subdir_counts in before_extraction.items():
        num_files_to_extract[class_name] = {}
        for subdir, count in subdir_counts.items():
            num_files_to_extract[class_name][subdir] = count // 10

    extracted_count = {}
    for file_path in random.sample(extracted_files, sum(sum(counts.values()) for counts in num_files_to_extract.values())):
        subdir = os.path.basename(os.path.dirname(file_path))
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        extracted_count.setdefault(class_name, {})
        extracted_count[class_name].setdefault(subdir, 0)
        if extracted_count[class_name][subdir] < num_files_to_extract[class_name][subdir]:
            extracted_count[class_name][subdir] += 1

    after_extraction = {}
    for class_name, subdir_counts in before_extraction.items():
        after_extraction[class_name] = {}
        for subdir, count in subdir_counts.items():
            after_extraction[class_name][subdir] = count - extracted_count[class_name].get(subdir, 0)

    return before_extraction, after_extraction

if __name__ == "__main__":
    zip_file = "/content/drive/MyDrive/Colab Notebooks/DeepFake Image Detection/DFDC.zip"  # Replace with the path to your zip file
    extract_dir = "/content/extracted_files"  # Replace with the directory where you want to extract files

    before_extraction, after_extraction = extract_files(zip_file, extract_dir)

    print("Initial count of files in each class before extraction:")
    for class_name, subdir_counts in before_extraction.items():
        print(f"{class_name}:")
        for subdir, count in subdir_counts.items():
            print(f"  {subdir}: {count}")

    print("\nCount of files after extracting 50% while preserving balance:")
    for class_name, subdir_counts in after_extraction.items():
        print(f"{class_name}:")
        for subdir, count in subdir_counts.items():
            print(f"  {subdir}: {count}")
