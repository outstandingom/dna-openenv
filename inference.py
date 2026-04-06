from knowledge_graph_env import KnowledgeGraphEnv
import sys

def main():
    env = KnowledgeGraphEnv()
    print("[START]")

    obs = env.reset()
    print(f"[STEP] Observation: {obs}")

    # Episode: 3 steps
    step = 0
    total_reward = 0.0
    while not env.state().get("step", 0) == 3:
        state = env.state()
        step_name = state["step_name"]
        # Simple agent heuristics based on observation
        if step_name == "identify":
            # Extract key concept from input (crude)
            words = obs.lower().split()
            if "login" in words or "account" in words:
                action = "login issue"
            elif "billing" in words or "payment" in words:
                action = "billing"
            elif "slow" in words or "performance" in words:
                action = "slow performance"
            elif "crash" in words:
                action = "crash"
            elif "feature" in words or "request" in words:
                action = "feature request"
            else:
                action = "hardware failure"
        elif step_name == "relate":
            # Use the identified concept to guess a relation
            if "login" in obs.lower() or "account" in obs.lower():
                action = "account locked"
            elif "billing" in obs.lower() or "payment" in obs.lower():
                action = "refund"
            elif "slow" in obs.lower():
                action = "crash"
            elif "feature" in obs.lower():
                action = "enhancement"
            else:
                action = "battery issue"
        else:  # answer
            if "login" in obs.lower() or "account" in obs.lower():
                action = "reset password"
            elif "billing" in obs.lower() or "payment" in obs.lower():
                action = "process refund"
            elif "slow" in obs.lower():
                action = "optimize code"
            elif "feature" in obs.lower():
                action = "implement new feature"
            else:
                action = "replace battery"

        obs, reward, done, info = env.step(action)
        print(f"[STEP] Action: {action}")
        print(f"[STEP] Reward: {reward:.4f}")
        total_reward += reward
        step += 1

    print(f"[END] Total reward: {total_reward:.4f}")
    env.close()

if __name__ == "__main__":
    main()