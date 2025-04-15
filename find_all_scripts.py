import os

def print_repo_structure_and_code(root_dir=".", exclude_dirs=None, exclude_files=None, output_file="repo_structure.txt"):
    """
    Saves the directory structure of the repo along with the content of JavaScript and TypeScript files.

    Args:
        root_dir (str): The root directory to start scanning. Defaults to the current directory.
        exclude_dirs (list): List of directories to exclude from scanning.
        exclude_files (list): List of files to exclude from scanning.
        output_file (str): The file to save the output.
    """
    if exclude_dirs is None:
        exclude_dirs = ["node_modules", "dist", "build", "venv", "__pycache__"]
    if exclude_files is None:
        exclude_files = [os.path.basename(__file__)]

    # Only include JavaScript and TypeScript files
    include_extensions = {".py"}

    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write("Repository Structure and Code:\n\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

            indent_level = dirpath.count(os.sep)
            out_file.write("  " * indent_level + f"[{os.path.basename(dirpath)}]\n")

            for filename in filenames:
                if filename in exclude_files:
                    continue

                filepath = os.path.join(dirpath, filename)
                file_ext = os.path.splitext(filename)[1]

                if file_ext in include_extensions:
                    out_file.write("  " * (indent_level + 1) + f"- {filename}\n")
                    try:
                        with open(filepath, "r", encoding="utf-8") as file:
                            out_file.write("  " * (indent_level + 2) + "[Code Start]\n")
                            for line in file:
                                out_file.write("  " * (indent_level + 2) + line)
                            out_file.write("  " * (indent_level + 2) + "[Code End]\n")
                    except Exception as e:
                        out_file.write("  " * (indent_level + 2) + f"[Error reading file: {e}]\n")

if __name__ == "__main__":
    print("Saving repository structure and code to repo_structure.txt...")
    print_repo_structure_and_code()
    print("Done! Check repo_structure.txt for output.")
