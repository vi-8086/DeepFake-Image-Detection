import zipfile
import os

def extract_files(zip_file, extract_dir):
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Check if the Dataset directory exists in the zip file
        if 'Dataset' not in zip_ref.namelist():
            print("Error: Dataset directory not found in the zip file.")
            return
        
        # Extract files from the Dataset directory
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith('Dataset/') and not file_info.is_dir():
                extracted_files.append(file_info.filename.split('/', 1)[1])  # Extract the file name without the 'Dataset/' prefix
                zip_ref.extract(file_info, extract_dir)
        print("Files extracted successfully.")
    return extracted_files

if __name__ == "__main__":
    zip_file = "DFDC.zip"  # Replace with the name of your zip file
    extract_dir = "extracted_files"  # Replace with the directory where you want to extract files
    extracted_files = extract_files(zip_file, extract_dir)
    print("Extracted files:", extracted_files)
