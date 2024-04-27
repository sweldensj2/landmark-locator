import os
import shutil

def make_nyc_dataset(
    download_path,
    train_path,
    val_path,
    label_dictionary,
    fraction_valid,
):

    # Clear existing contents of train and val folders
    clear_folder(train_path)
    clear_folder(val_path)


    # for every folder in the downloaded images path
    for folder_name in os.listdir(download_path):
        # Check if the item is a directory
        if folder_name == ".ipynb_checkpoints":
            continue

        # if we are dealing with a folder
        if os.path.isdir(os.path.join(download_path, folder_name)):
            # Process the folder here, for example:
            print("Processing folder:", folder_name)
            folder_path = os.path.join(download_path, folder_name)

            #get all text files within this folder
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            num_valid_txt_files = int(len(txt_files) * fraction_valid) #rounds how many we should do based on the fraction
            num_processed_txt_files = 0

            #for all text files
            for file_name in txt_files: 
                if num_processed_txt_files < num_valid_txt_files:
                    # Processing for validation set
                    process_file(file_name, folder_path, val_path, label_dictionary)
                    num_processed_txt_files += 1
                else:
                    # Processing for training set
                    process_file(file_name, folder_path, train_path, label_dictionary)


# def process_file(file_name, folder_path, dest_path, label_dictionary):
#     #get the raw name of the text file
#     raw_name = os.path.splitext(file_name)[0]
#     print("Raw Name:", raw_name)
#     # get the name of the building folder and create a combo name
#     folder_name = os.path.basename(folder_path)
#     combo_name = folder_name + "_" + raw_name
#     print("Combo Name:", combo_name)

#     # Copy the .txt file to the destination folder with the new combo name
#     src_txt_file = os.path.join(folder_path, file_name)
#     dest_txt_file = os.path.join(dest_path, combo_name + ".txt")
#     shutil.copy(src_txt_file, dest_txt_file)

#     # Copy the corresponding .jpg file to the destination folder with the new combo name
#     src_jpg_file = os.path.join(folder_path, raw_name + ".jpg")
#     dest_jpg_file = os.path.join(dest_path, combo_name + ".jpg")
#     shutil.copy(src_jpg_file, dest_jpg_file)

def process_file(file_name, folder_path, dest_path, label_dictionary):
    # Get the raw name of the text file
    raw_name = os.path.splitext(file_name)[0]
    print("Raw Name:", raw_name)
    # Get the name of the building folder and create a combo name
    folder_name = os.path.basename(folder_path)
    combo_name = folder_name + "_" + raw_name
    print("Combo Name:", combo_name)

    # Check if the corresponding .jpg file exists
    src_jpg_file = os.path.join(folder_path, raw_name + ".jpg")
    if os.path.exists(src_jpg_file):
        # Copy the .txt file to the destination folder with the new combo name
        src_txt_file = os.path.join(folder_path, file_name)
        dest_txt_file = os.path.join(dest_path, combo_name + ".txt")
        shutil.copy(src_txt_file, dest_txt_file)
        
        # Copy the .jpg file to the destination folder with the new combo name
        dest_jpg_file = os.path.join(dest_path, combo_name + ".jpg")
        shutil.copy(src_jpg_file, dest_jpg_file)
    else:
        print(f"Skipping copying of {combo_name} as corresponding .jpg does not exist.")


def clear_folder(folder_path):
    # Clear existing contents of a folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")