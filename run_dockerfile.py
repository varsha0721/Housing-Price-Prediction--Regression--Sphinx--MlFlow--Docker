import subprocess

# Run your custom script or tasks here
print("Running all three scripts...")

subprocess.run(["python3", "ingest_data.py"])
subprocess.run(["python3", "train.py"])
subprocess.run(["python3", "score.py"])

print("All scripts successfully ran...")

# Start MLflow
print("Running MLflow...")
subprocess.run(["mlflow", "server", "--host", "0.0.0.0"])

# Access MLflow UI
print("Access MLflow UI at http://localhost:5000")
