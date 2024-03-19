import zipfile
import os

def extract_files(zip_file, extract_dir):
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all files and directories from the zip file
        zip_ref.extractall(extract_dir)
        # Iterate over the extracted files and directories
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                # Remove the prefix "Dataset/" from the file path
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace('Dataset/', '')
                extracted_files.append(new_file_path)
        print("Files extracted successfully.")
    return extracted_files

if __name__ == "__main__":
    zip_file = "/content/drive/MyDrive/Colab Notebooks/DeepFake Image Detection/DFDC.zip"  # Replace with the path to your zip file
    extract_dir = "/content/extracted_files"  # Replace with the directory where you want to extract files
    extracted_files = extract_files(zip_file, extract_dir)
    print("Extracted files:", extracted_files)




