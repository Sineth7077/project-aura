"""Script to execute notebooks 05-08 with fresh kernels."""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(notebook_path, output_path):
    """Execute a notebook and save the result."""
    print(f"Executing {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"Successfully executed and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")
        return False

if __name__ == "__main__":
    notebooks_dir = "notebooks"
    
    notebooks_to_execute = [
        ("05_gesture_recognition.ipynb", "05_gesture_recognition_executed.ipynb"),
        ("06_ai_agents.ipynb", "06_ai_agents_executed.ipynb"),
        ("07_deployment_prep.ipynb", "07_deployment_prep_executed.ipynb"),
        ("08_monitoring_analytics.ipynb", "08_monitoring_analytics_executed.ipynb"),
    ]
    
    results = []
    for input_name, output_name in notebooks_to_execute:
        input_path = os.path.join(notebooks_dir, input_name)
        output_path = os.path.join(notebooks_dir, output_name)
        
        if os.path.exists(input_path):
            success = execute_notebook(input_path, output_path)
            results.append((input_name, success))
        else:
            print(f"Notebook not found: {input_path}")
            results.append((input_name, False))
    
    print("\n=== Execution Summary ===")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{name}: {status}")
