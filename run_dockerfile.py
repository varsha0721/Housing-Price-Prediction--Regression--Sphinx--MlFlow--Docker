import subprocess

# Run your custom script or tasks here
print("Running driver script for parent child run...")

subprocess.run(["python3", "driver.py"])
print("Driver scripts successfully ran...")

# Start MLflow
print("Running MLflow...")
subprocess.run(["mlflow", "server", "--host", "0.0.0.0"])

# Access MLflow UI
print("Access MLflow UI at http://localhost:5000")
