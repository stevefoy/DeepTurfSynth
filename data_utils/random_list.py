import os 
import random




def read_file_into_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def write_list_to_file(file_path, input_list):
    try:
        with open(file_path, 'w') as file:
            for item in input_list:
                file.write(str(item)+"\r\n")
        print("Successfully.")
    except IOError:
        print(f"Error writing")


from os import walk


# Could load this via arg
path = "C:\\Users\\stevf\\Documents\\Projects\\datasets\\Turfgrass\\full_resolution\\"
mypath = "C:\\Users\\stevf\\Documents\\Projects\\datasets\\Turfgrass\\full_resolution\\Images"
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

filename = "C:\\Users\\stevf\\Documents\\Projects\\datasets\\Turfgrass\\full_resolution\\FullImageList.txt"
outfile = open(filename, 'w')
outfile.writelines([str(i)+'\n' for i in f])
outfile.close()


file_path = 'FullImageList.txt'  # Replace 'example.txt' with the path to your file
file_val_path = 'ImagesValidation.txt'  # Replace 'example.txt' with the path to your file
file_train_path = 'ImagesTrain.txt'  # Replace 'example.txt' with the path to your file
file_test_path = 'ImagesTest.txt'  # Replace 'example.txt' with the path to your file

full_images_path = os.path.join(path, file_path)

fill_image_list = read_file_into_list(full_images_path  )

# 20 percent of the data
ratio_count = int(len(fill_image_list)*0.1)

selected_test_items = random.sample(fill_image_list, ratio_count)

remaining_items = list(set(fill_image_list) - set(selected_test_items))

selected_val_items = random.sample(remaining_items, ratio_count)

remaining_items = list(set(remaining_items) - set(selected_val_items))

write_list_to_file(os.path.join(path, file_test_path), selected_test_items)
write_list_to_file(os.path.join(path, file_val_path), selected_val_items)
write_list_to_file(os.path.join(path, file_train_path), remaining_items)




