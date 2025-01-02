import torch
import os

def inspect_model(model_path='data/transformer.pth'):
    print(f"\n=== Inspecting model: {model_path} ===")
    
    try:
        # Try loading the model with lambda function
        state_dict = torch.load(model_path, map_location=lambda _s, _: _s)
        
        # Let's print the device of each parameter to understand mapping
        print("\nModel layers and their devices:")
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{key}: device {tensor.device}")
                
        # The lambda function lambda _s, _: _s works like this:
        # _s: source device (where tensor was saved)
        # _: destination device (where we want to load)
        # returning _s means "keep tensor on its original device"
        
        first_param = next(iter(state_dict.values()))
        print(f"\nOriginal save device: {first_param.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model('data/transformer.pth') 