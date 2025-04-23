import torch


def main():
    print("Hello from all-ai-tutorials!")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available (NVIDIA GPU): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not found by PyTorch.")


if __name__ == "__main__":
    main()
