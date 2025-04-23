#!/usr/bin/env python3
import sys
import importlib.metadata  # To safely get versions

print("--- Checking Essential Libraries ---")

libraries_ok = True
found_versions = {}

# List of libraries to check (use the name you import with)
# We'll map to the install name if needed for version checking
libraries_to_check = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",  # Note: import name is 'sklearn', install name is 'scikit-learn'
    "torch": "torch",
}

for import_name, install_name in libraries_to_check.items():
    try:
        # Try importing the library
        __import__(import_name)
        # If import succeeds, try to get the version
        try:
            version = importlib.metadata.version(install_name)
            found_versions[import_name] = version
            print(f"[ OK ] Imported '{import_name}' successfully (Version: {version})")
        except importlib.metadata.PackageNotFoundError:
            # Should not happen if import succeeded, but just in case
            found_versions[import_name] = "unknown"
            print(f"[ OK ] Imported '{import_name}' successfully (Version: unknown)")

    except ImportError:
        print(f"[FAIL] Could not import '{import_name}'. Please install '{install_name}'.")
        libraries_ok = False
    except Exception as e:
        # Catch any other unexpected error during import
        print(f"[ERROR] An error occurred while trying to import '{import_name}': {e}")
        libraries_ok = False

print("------------------------------------")

# Specific check for PyTorch CUDA, only if PyTorch was imported
if "torch" in found_versions:
    print("\n--- Checking PyTorch CUDA Support ---")
    try:
        # We know torch is imported, so we can use it directly
        import torch

        # Use the previously found version for consistency
        print(f"PyTorch Version: {found_versions['torch']}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            # Only get device name if count > 0
            if torch.cuda.device_count() > 0:
                print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA reported available, but no devices found.")  # Edge case
        else:
            print("CUDA not detected or not supported by this PyTorch build.")
            # Add a hint for CPU builds
            if "+cpu" in found_versions["torch"]:
                print("Hint: The installed PyTorch version appears to be CPU-only.")

    except Exception as e:
        print(f"[ERROR] An error occurred during PyTorch CUDA check: {e}")
        libraries_ok = False  # If CUDA check fails, flag it
    print("------------------------------------")
elif libraries_ok:
    # This case shouldn't normally be hit if torch is in the check list
    # but handles if it was removed from the list above.
    print("\nSkipping PyTorch CUDA check because PyTorch import failed or wasn't checked.")


# Final Summary
print("\n--- Summary ---")
if libraries_ok:
    print("All essential libraries imported successfully!")
    print("Environment looks good to go for the tutorial.")
    sys.exit(0)  # Exit with success code
else:
    print("One or more libraries failed to import or check. Please review the messages above.")
    print("Install missing packages (e.g., using 'uv pip install <package_name>').")
    sys.exit(1)  # Exit with error code

# Note: This script doesn't enforce specific minimum versions,
# it just checks if the libraries can be imported.
