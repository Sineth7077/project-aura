#!/usr/bin/env python3
"""Validate all notebooks and source files in Project Aura 2.0."""

import json
import os
import sys
import py_compile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, "notebooks")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
TESTS_DIR = os.path.join(PROJECT_DIR, "tests")

NOTEBOOKS = [
    "01_environment_setup.ipynb",
    "02_data_ingestion.ipynb",
    "03_model_training.ipynb",
    "04_inference_engine.ipynb",
    "05_gesture_recognition.ipynb",
    "06_ai_agents.ipynb",
    "07_deployment_prep.ipynb",
    "08_monitoring_analytics.ipynb",
]


def validate_notebook(path):
    """Validate a single .ipynb file. Returns (is_valid, error_list)."""
    errors = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    if nb.get("nbformat") != 4:
        errors.append(f"Invalid nbformat: {nb.get('nbformat')}")

    cells = nb.get("cells", [])
    if not cells:
        errors.append("No cells found")

    for i, cell in enumerate(cells):
        if "cell_type" not in cell:
            errors.append(f"Cell {i}: missing cell_type")
        if cell.get("cell_type") == "code":
            if "execution_count" not in cell:
                errors.append(f"Cell {i}: code cell missing execution_count")
            if "outputs" not in cell:
                errors.append(f"Cell {i}: code cell missing outputs")
        if "source" not in cell:
            errors.append(f"Cell {i}: missing source")

    if "metadata" not in nb:
        errors.append("Missing metadata")
    if "kernelspec" not in nb.get("metadata", {}):
        errors.append("Missing kernelspec")

    return len(errors) == 0, errors


def compile_python_file(path):
    """Try to compile a Python file. Returns (is_valid, error_message)."""
    try:
        py_compile.compile(path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("Project Aura 2.0 - Validation Suite")
    print("=" * 60)
    print()

    # Validate notebooks
    print("Validating Google Colab Notebooks:")
    print("-" * 40)
    all_nb_valid = True
    for nb_name in NOTEBOOKS:
        path = os.path.join(NOTEBOOKS_DIR, nb_name)
        if not os.path.exists(path):
            print(f"  FAIL: {nb_name} - file not found")
            all_nb_valid = False
            continue
        is_valid, errors = validate_notebook(path)
        if is_valid:
            with open(path) as f:
                nb = json.load(f)
            cells = nb.get("cells", [])
            md_count = sum(1 for c in cells if c.get("cell_type") == "markdown")
            code_count = sum(1 for c in cells if c.get("cell_type") == "code")
            print(f"  OK:   {nb_name} - {len(cells)} cells ({md_count}md, {code_count}code)")
        else:
            print(f"  FAIL: {nb_name}")
            for e in errors:
                print(f"          - {e}")
            all_nb_valid = False
    print()

    # Validate Python source files
    print("Validating Python Source Files:")
    print("-" * 40)
    all_py_valid = True
    py_files = []
    for root, dirs, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    for root, dirs, files in os.walk(TESTS_DIR):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    for fpath in sorted(py_files):
        rel = os.path.relpath(fpath, PROJECT_DIR)
        is_valid, error = compile_python_file(fpath)
        if is_valid:
            print(f"  OK:   {rel}")
        else:
            print(f"  FAIL: {rel} - {error}")
            all_py_valid = False
    print()

    # Summary
    print("=" * 60)
    if all_nb_valid and all_py_valid:
        print("RESULT: ALL VALIDATIONS PASSED")
        print(f"  Notebooks: {len(NOTEBOOKS)}/{len(NOTEBOOKS)} valid")
        print(f"  Python files: {len(py_files)}/{len(py_files)} compile")
        return 0
    else:
        print("RESULT: VALIDATION FAILED")
        if not all_nb_valid:
            print("  Some notebooks have errors")
        if not all_py_valid:
            print("  Some Python files have errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())