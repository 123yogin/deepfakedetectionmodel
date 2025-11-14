"""
Test script to verify Phase 1 project skeleton is correctly set up.
Tests directory structure, imports, and basic project organization.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_directory_structure():
    """Test that all required directories exist."""
    required_dirs = [
        'backend',
        'models',
        'workers',
        'training',
        'frontend',
        'storage',
        'results',
        'scripts',
        'docker',
        'tests'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        elif not dir_path.is_dir():
            missing_dirs.append(f"{dir_name} (exists but is not a directory)")
    
    assert not missing_dirs, f"Missing required directories: {missing_dirs}"
    print("[PASS] All required directories exist")


def test_init_files():
    """Test that all __init__.py files exist and are importable."""
    modules_to_test = [
        'backend',
        'models',
        'workers',
        'training',
        'frontend',
        'storage',
        'results',
        'scripts',
        'docker',
        'tests'
    ]
    
    failed_imports = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"[PASS] Successfully imported {module_name}")
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
    
    assert not failed_imports, f"Failed to import modules: {failed_imports}"
    print("[PASS] All modules are importable")


def test_readme_exists():
    """Test that README.md exists."""
    readme_path = project_root / 'README.md'
    assert readme_path.exists(), "README.md does not exist"
    assert readme_path.is_file(), "README.md exists but is not a file"
    print("[PASS] README.md exists")


def test_init_file_contents():
    """Test that __init__.py files have appropriate docstrings."""
    modules_with_docs = {
        'backend': 'Backend API server',
        'models': 'ML models',
        'workers': 'Background workers',
        'tests': 'Test suite'
    }
    
    for module_name, expected_keyword in modules_with_docs.items():
        init_path = project_root / module_name / '__init__.py'
        if init_path.exists():
            content = init_path.read_text(encoding='utf-8')
            assert expected_keyword.lower() in content.lower(), \
                f"{module_name}/__init__.py should mention '{expected_keyword}'"
            print(f"[PASS] {module_name}/__init__.py has appropriate documentation")


def test_project_structure_matches_readme():
    """Test that the actual structure matches README description."""
    readme_path = project_root / 'README.md'
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding='utf-8')
        
        # Check that README mentions key directories
        key_dirs = ['backend', 'models', 'workers', 'frontend', 'tests']
        for dir_name in key_dirs:
            assert dir_name in readme_content, \
                f"README.md should mention '{dir_name}' directory"
        
        print("[PASS] Project structure matches README description")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("Testing Phase 1: Project Skeleton")
    print("="*60 + "\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("__init__.py Files", test_init_files),
        ("README.md Exists", test_readme_exists),
        ("__init__.py Documentation", test_init_file_contents),
        ("Structure Matches README", test_project_structure_matches_readme),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}: Unexpected error - {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("[SUCCESS] Phase 1 project skeleton is correctly set up!")
        return True
    else:
        print("[WARNING] Some tests failed. Please review the issues above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

