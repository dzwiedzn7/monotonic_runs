import os


def rename_files_in_directory(directory):
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort the files alphabetically
    files.sort()

    # Rename each file
    for i, filename in enumerate(files, start=1):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]

        # Create the new filename
        new_filename = f"{i}{file_extension}"

        # Get full file paths
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")


# Specify the directory
directory = 'rr'

# Call the function
rename_files_in_directory(directory)
