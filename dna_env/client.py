import os
import sys
import requests
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860")  # Your HF Space URL

TASK_NAME = "knowledge_graph"
BENCHMARK = "knowledge_graph_env"
MAX_STEPS = 10

def main():
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})
    
    # Reset environment
    resp = session.post(f"{SPACE_URL}/reset")
    obs = resp.json()["observation"]
    
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
    
    step = 0
    rewards = []
    success = False
    
    while step < MAX_STEPS:
        # Simple heuristic agent
        words = obs.lower().split()
        if "login" in words or "account" in words:
            action = "login issue"
        elif "billing" in words or "payment" in words:
            action = "billing"
        elif "slow" in words or "performance" in words:
            action = "slow performance"
        elif "crash" in words:
            action = "crash"
        else:
            action = "hardware failure"
        
        # Take step
        resp = session.post(f"{SPACE_URL}/step", json={"action": action})
        data = resp.json()
        
        reward = data["reward"]
        done = data["done"]
        rewards.append(reward)
        
        print(f"[STEP] step={step+1} action={action} reward={reward:.2f} done={str(done).lower()} error=null")
        
        obs = data["observation"]
        step += 1
        
        if done:
            success = True
            break
    
    # Get final state
    resp = session.get(f"{SPACE_URL}/state")
    state = resp.json()["state"]
    
    # Calculate score (average reward)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    main()
