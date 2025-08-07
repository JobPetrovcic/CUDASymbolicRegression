import torch

# Function that gets a CUDA device index with at least 1 GB of memory
def get_cuda_device_with_min_memory(min_memory_gb: int = 1) -> int:
    min_memory_bytes = min_memory_gb * 1024**3
    for i in range(torch.cuda.device_count()):
        try:
            free_mem, _ = torch.cuda.mem_get_info(i)
            if free_mem >= min_memory_bytes:
                return i
        except Exception as e:
            continue
        
    raise RuntimeError(f"No CUDA device with at least {min_memory_gb} GB of memory found.")

if __name__ == "__main__":
    print(get_cuda_device_with_min_memory(1))