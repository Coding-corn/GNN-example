import os
import shutil

# Specify the directory to clean
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Specify the list of files and folders to keep
# TODO Revert
# keep_list = ["cleanup.py", "main.py", "README.md"]
keep_list = ["cleanup.py", "main.py", "README.md", "packages"]

# Get absolute paths of items to keep
keep_paths = [os.path.join(THIS_FOLDER, item) for item in keep_list]

# Iterate through all items in the directory
for item in os.listdir(THIS_FOLDER):
    item_path = os.path.join(THIS_FOLDER, item)

    # Skip items in the keep list
    if item_path in keep_paths:
        continue

    # Delete files and directories
    if os.path.isfile(item_path):
        os.remove(item_path)
        print(f"Deleted file: {item_path}")
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)
        print(f"Deleted folder: {item_path}")

print("Cleanup complete.")