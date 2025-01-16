import ast
import os
import subprocess

def extract_imports_from_file(file_path):
    """Extract imports from a single Python file."""
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])  # Get top-level module
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def extract_imports_from_directory(directory):
    """Extract imports from all Python files in a directory."""
    all_imports = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Process only Python files
                file_path = os.path.join(root, file)
                imports = extract_imports_from_file(file_path)
                all_imports.update(imports)
    return all_imports

def map_to_packages(modules):
    """Map modules to installable packages using pip show."""
    packages = set()
    for module in modules:
        try:
            # Check if the module corresponds to an installed package
            result = subprocess.run(
                ["pip", "show", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if result.stdout:
                packages.add(module)
        except Exception:
            pass
    return packages

if __name__ == "__main__":
    project_directory = r"C:\Users\62812\Documents\Kuliah\Semester 4\Daftar kerja\Sawit Pro\code\final assignment\object-detection"  # Replace with your project's root directory
    if os.path.exists(project_directory):
        imports = extract_imports_from_directory(project_directory)
        packages = map_to_packages(imports)
        
        with open("requirements.txt", "w") as req_file:
            for package in sorted(packages):  # Sort packages alphabetically
                req_file.write(f"{package}\n")
        print("requirements.txt created successfully!")
    else:
        print(f"Directory {project_directory} not found.")
