import torch


def main():
    print("Hello from all-ai-tutorials!")


if __name__ == "__main__":
    main()
    print(torch.__version__)
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    print(torch.cuda.is_available())
