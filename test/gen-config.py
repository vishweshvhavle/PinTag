import json
import random

def generate_configs():
    configs = []
    
    for i in range(200):
        config = {
            "theta": random.uniform(-180, 180),
            "phi": random.uniform(20, 90),
            "radius": random.uniform(1.6, 7.0)
        }
        
        if i >= 150:
            config.update({
                "kernel_size": [random.randint(2, 7)] * 2,
                "sigma": random.uniform(1.0, 1.6),
                "grain_intensity": random.uniform(0, 0.4),
                "grain_density": random.uniform(0, 0.7)
            })
        
        configs.append(config)
    
    return configs

def main():
    configs = generate_configs()
    
    with open('frame_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated 200 frame configurations and saved to 'frame_configs.json'")

if __name__ == "__main__":
    main()