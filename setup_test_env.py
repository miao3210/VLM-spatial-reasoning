#!/usr/bin/env python3
"""
Automated test environment setup script.
Scans all test files, extracts dependencies, and installs required packages.
"""

import ast
import sys
import subprocess
from pathlib import Path
from typing import Set, Dict

# Mapping from import names to pip package names
IMPORT_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
    'google': 'google-generativeai',  # google.generativeai package
    'anthropic': 'anthropic',
    'openai': 'openai',
    'transformers': 'transformers',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'accelerate': 'accelerate',
    'qwen_vl_utils': 'qwen-vl-utils',
}

# Standard library modules that don't need installation
STDLIB_MODULES = {
    'abc', 'argparse', 'base64', 'collections', 'contextlib', 'copy', 'datetime',
    'functools', 'io', 'json', 'math', 'os', 'pathlib', 're', 'sys', 'time',
    'typing', 'warnings', 'unittest', 'random', 'itertools', 'operator',
}


class ImportExtractor(ast.NodeVisitor):
    """Extract all import statements from Python AST."""
    
    def __init__(self):
        self.imports: Set[str] = set()
    
    def visit_Import(self, node):
        """Handle 'import x' statements."""
        for alias in node.names:
            # Get the top-level module name
            module = alias.name.split('.')[0]
            self.imports.add(module)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Handle 'from x import y' statements."""
        if node.module:
            # Get the top-level module name
            module = node.module.split('.')[0]
            self.imports.add(module)
        self.generic_visit(node)


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        extractor = ImportExtractor()
        extractor.visit(tree)
        return extractor.imports
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()


def get_all_imports(test_dir: Path) -> Set[str]:
    """Scan all Python files in test directory and extract imports."""
    all_imports = set()
    
    for py_file in test_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue  # Skip __init__.py, etc.
        
        print(f"Scanning {py_file.name}...")
        imports = extract_imports_from_file(py_file)
        all_imports.update(imports)
    
    return all_imports


def filter_third_party_imports(imports: Set[str]) -> Set[str]:
    """Filter out stdlib and local package imports."""
    third_party = set()
    
    for imp in imports:
        # Skip stdlib modules
        if imp in STDLIB_MODULES:
            continue
        
        # Skip local package (spatok)
        if imp == 'spatok':
            continue
        
        third_party.add(imp)
    
    return third_party


def map_imports_to_packages(imports: Set[str]) -> Dict[str, str]:
    """Map import names to pip package names."""
    packages = {}
    
    for imp in imports:
        # Check if there's a custom mapping
        if imp in IMPORT_TO_PACKAGE:
            packages[imp] = IMPORT_TO_PACKAGE[imp]
        else:
            # Default: assume package name matches import name
            packages[imp] = imp
    
    return packages


def install_packages(packages: Dict[str, str], conda_env: str = 'vlm'):
    """Install packages using pip in specified conda environment."""
    if not packages:
        print("\nNo packages to install!")
        return
    
    print(f"\n{'='*60}")
    print(f"PACKAGES TO INSTALL IN CONDA ENV: {conda_env}")
    print('='*60)
    for imp, pkg in sorted(packages.items()):
        print(f"  {imp:20} -> {pkg}")
    print('='*60)
    
    # Get unique package names
    unique_packages = sorted(set(packages.values()))
    
    # Special handling for PyTorch - install with CUDA support
    torch_packages = []
    other_packages = []
    
    for pkg in unique_packages:
        if pkg in ['torch', 'torchvision', 'torchaudio']:
            torch_packages.append(pkg)
        else:
            other_packages.append(pkg)
    
    # Install PyTorch with CUDA support first
    if torch_packages:
        print(f"\nInstalling PyTorch packages with CUDA 12.1 support in {conda_env} env...")
        print(f"Packages: {', '.join(torch_packages)}")
        cmd = [
            'conda', 'run', '-n', conda_env, 'pip', 'install', '-v',
            '--index-url', 'https://download.pytorch.org/whl/cu121',
        ] + torch_packages
        
        print(f"\nCommand: {' '.join(cmd)}")
        print("This may take several minutes...\n")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nWarning: PyTorch installation failed with code {result.returncode}")
        else:
            print(f"\n✓ PyTorch installation successful")
    
    # Install other packages
    if other_packages:
        print(f"\nInstalling {len(other_packages)} additional packages in {conda_env} env...")
        print(f"Packages: {', '.join(other_packages)}")
        cmd = ['conda', 'run', '-n', conda_env, 'pip', 'install', '-v'] + other_packages
        
        print(f"\nCommand: {' '.join(cmd)}")
        print("This may take a few minutes...\n")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nWarning: Package installation failed with code {result.returncode}")
        else:
            print(f"\n✓ Additional packages installation successful")
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE")
    print("="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("AUTOMATED TEST ENVIRONMENT SETUP")
    print("="*60)
    
    # Find test directory and package source directory
    base_dir = Path(__file__).parent
    test_dir = base_dir / 'spatok' / 'test'
    vlms_dir = base_dir / 'spatok' / 'vlms'
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    print(f"\nScanning test directory: {test_dir}")
    print(f"Scanning VLMs directory: {vlms_dir}\n")
    
    # Step 1: Extract all imports from test files and package files
    all_imports = get_all_imports(test_dir)
    print(f"Found {len(all_imports)} imports from test files")
    
    if vlms_dir.exists():
        vlms_imports = get_all_imports(vlms_dir)
        print(f"Found {len(vlms_imports)} imports from VLMs package")
        all_imports.update(vlms_imports)
    
    print(f"\nTotal: {len(all_imports)} imports")
    
    # Step 2: Filter third-party imports
    third_party = filter_third_party_imports(all_imports)
    print(f"Found {len(third_party)} third-party imports")
    
    # Step 3: Map to package names
    packages = map_imports_to_packages(third_party)
    
    # Step 4: Install packages
    install_packages(packages)
    
    print("\n✓ Setup complete! You can now run your tests.")


if __name__ == '__main__':
    main()
