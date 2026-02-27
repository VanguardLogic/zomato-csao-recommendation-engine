# run_full_pipeline.py

import subprocess
import sys
import os

def run_step(command, description):
    print(f"\n>>> STEP: {description}")
    print(f"Executing: {command}")
    try:
        # Using sys.executable to ensure we use the same python environment
        process = subprocess.Popen([sys.executable] + command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(f"  {line.strip()}")
        process.wait()
        if process.returncode != 0:
            print(f"❌ ERROR in {description}")
            return False
        print(f"✅ SUCCESS: {description} completed.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

def main():
    print("====================================================")
    print("   ZOMATO CSAO HACKATHON - FULL PIPELINE RUNNER     ")
    print("====================================================")

    # 1. Setup/Verify Directories
    os.makedirs("data", exist_ok=True)

    # 2. Sequence of Operations
    steps = [
        ("1_Model_Development/data_prep/generate_synthetic_data.py", "Generating 15k Synthetic Orders"),
        ("1_Model_Development/offline_pipeline/build_graph.py", "Building regional Knowledge Graph"),
        ("1_Model_Development/offline_pipeline/train_ranker.py", "Training Two-Stage ML Ranker"),
        ("2_Evaluation_Results/metrics.py", "Running Performance Evaluation")
    ]

    for script, desc in steps:
        if not run_step(script, desc):
            sys.exit(1)

    print("\n====================================================")
    print("🎉 FULL PIPELINE COMPLETE!")
    print("====================================================")
    print("You can now start the API servers:")
    print("  Port 8001 (God-Mode): python api/app.py")
    print("  Port 8002 (Two-Stage ML): python api/app_v2.py")
    print("====================================================")

if __name__ == "__main__":
    main()
