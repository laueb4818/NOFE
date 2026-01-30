
import torch
from utilities import estimate_global_Z

def check_grad():
    output = torch.randn(100, 16, requires_grad=True)
    try:
        Z = estimate_global_Z(output)
        print(f"Z requires_grad: {Z.requires_grad}")
        print(f"Z grad_fn: {Z.grad_fn}")
        
        # Try backward
        try:
            Z.backward()
            print("Backward successful")
            print(f"Output grad norm: {output.grad.norm().item()}")
        except Exception as e:
            print(f"Backward failed: {e}")
            
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    check_grad()
