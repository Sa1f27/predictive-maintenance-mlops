# debug_imports.py - Diagnose import issues
import os
import sys
import importlib.util

def check_file_exists(file_path):
    """Check if file exists and print its size"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {file_path} exists ({size} bytes)")
        return True
    else:
        print(f"❌ {file_path} MISSING")
        return False

def check_class_in_file(file_path, class_name):
    """Check if a class exists in a Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if f"class {class_name}" in content:
                print(f"✅ Class '{class_name}' found in {file_path}")
                return True
            else:
                print(f"❌ Class '{class_name}' NOT found in {file_path}")
                # Show what classes are actually there
                lines = content.split('\n')
                classes = [line.strip() for line in lines if line.strip().startswith('class ')]
                if classes:
                    print(f"   Available classes: {classes}")
                else:
                    print("   No classes found in file")
                return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def try_import(module_path, class_name):
    """Try to import a class and show the error"""
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            print(f"✅ Successfully imported {class_name} from {module_path}")
            return True
        else:
            print(f"❌ {class_name} not found in {module_path}")
            available = [attr for attr in dir(module) if not attr.startswith('_')]
            print(f"   Available: {available}")
            return False
    except Exception as e:
        print(f"❌ Import error for {module_path}: {e}")
        return False

def main():
    print("🔍 DEBUGGING IMPORT ISSUES")
    print("="*60)
    
    # Check required files exist
    print("\n📁 CHECKING FILE EXISTENCE:")
    files_to_check = [
        "src/__init__.py",
        "src/components/__init__.py", 
        "src/components/data_ingestion.py",
        "src/components/data_transformation.py",
        "src/components/model_trainer.py",
        "src/pipeline/__init__.py",
        "src/pipeline/train_pipeline.py",
        "src/pipeline/predict_pipeline.py",
        "src/utils.py",
        "src/exception.py",
        "src/logger.py"
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not check_file_exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Create these missing files: {missing_files}")
    
    # Check classes exist in files
    print("\n🔍 CHECKING CLASS DEFINITIONS:")
    class_checks = [
        ("src/components/data_ingestion.py", "DataIngestion"),
        ("src/components/data_transformation.py", "DataTransformation"), 
        ("src/components/model_trainer.py", "ModelTrainer"),
    ]
    
    for file_path, class_name in class_checks:
        if os.path.exists(file_path):
            check_class_in_file(file_path, class_name)
    
    # Try actual imports
    print("\n🔧 TESTING IMPORTS:")
    import_tests = [
        ("src.components.data_ingestion", "DataIngestion"),
        ("src.components.data_transformation", "DataTransformation"),
        ("src.components.model_trainer", "ModelTrainer"),
    ]
    
    for module_path, class_name in import_tests:
        try_import(module_path, class_name)
    
    print("\n" + "="*60)
    print("🎯 QUICK FIXES:")
    print("1. Create missing __init__.py files:")
    print("   touch src/__init__.py")
    print("   touch src/components/__init__.py") 
    print("   touch src/pipeline/__init__.py")
    print("   touch src/mlops/__init__.py")
    print("\n2. Check class names match exactly in your files")
    print("3. Make sure no syntax errors in Python files")
    print("="*60)

if __name__ == "__main__":
    main()