import json
import random

def generate_radius_configs():
    configs = []
    for radius in range(1, 11):
        for _ in range(10):
            config = {
                "theta": random.uniform(-180, 180),
                "phi": 90,
                "radius": float(radius)
            }
            configs.append(config)
    return configs

def generate_phi_configs():
    configs = []
    for phi in range(0, 91):
        for _ in range(5):
            config = {
                "theta": random.uniform(-180, 180),
                "phi": float(phi),
                "radius": 2.0
            }
            configs.append(config)
    return configs

def generate_random_configs():
    configs = []
    for _ in range(3):
        config = {
            "theta": random.uniform(-180, 180),
            "phi": random.uniform(30, 90),
            "radius": random.uniform(1, 2)
        }
        configs.append(config)
    return configs

def main():
    # radius_configs = generate_radius_configs()
    # with open('radius_frame_configs.json', 'w') as f:
    #     json.dump(radius_configs, f, indent=2)
    # print(f"Generated {len(radius_configs)} radius frame configurations and saved to 'radius_frame_configs.json'")
    
    # phi_configs = generate_phi_configs()
    # with open('phi_frame_configs.json', 'w') as f:
    #     json.dump(phi_configs, f, indent=2)
    # print(f"Generated {len(phi_configs)} phi frame configurations and saved to 'phi_frame_configs.json'")

    random_configs = generate_random_configs()  
    with open('random_frame_configs.json', 'w') as f:
        json.dump(random_configs, f, indent=2)

if __name__ == "__main__":
    main()
