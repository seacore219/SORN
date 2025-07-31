import time

print("Starting tables import check...")
start_time = time.time()

try:
    import tables
    import_time = time.time() - start_time
    
    print(f" Tables imported successfully in {import_time:.2f} seconds")
    print(f"Tables version: {tables.__version__}")
    print(f"Tables location: {tables.__file__}")
    
    # Check if there are any obvious issues
    print(f"Tables HDF5 version: {tables.hdf5_version}")
    print(f"Tables is_pytables_file working: {hasattr(tables, 'is_pytables_file')}")
    
except ImportError as e:
    print(f" Failed to import tables: {e}")
except Exception as e:
    print(f" Error during tables import: {e}")

print("Check completed.")