import os
import shutil
import random
import hashlib
from collections import Counter

def calculate_sha256(filepath):
    """Calculate SHA-256 of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def prepare_splits(dataset_dir, output_dir, seed=42):
    print(f"Loading dataset from: {dataset_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    classes = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]
    
    # Initialize split directories
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
            
    # Track files and their hashes for split leakage verification
    split_hashes = {split: set() for split in splits}
    split_counts = {split: Counter() for split in splits}
    
    # For cross-split duplicate checks
    all_file_hashes = {} # hash -> (original_path, assigned_split)
    leakage_detected = False
    
    # Loop over each class to perform stratified split
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Error: Directory not found for class {cls} at {cls_dir}")
            return False
            
        # Get sorted list of files for reproducibility
        files = sorted([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        
        # Shuffle with fixed seed
        rng = random.Random(seed)
        rng.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        # Remaining goes to test to ensure we account for rounding errors and utilize all images
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        assigned_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        for split, split_files in assigned_splits.items():
            for f in split_files:
                src_path = os.path.join(cls_dir, f)
                dst_path = os.path.join(output_dir, split, cls, f)
                
                # Check hash for duplicate validation
                fhash = calculate_sha256(src_path)
                
                if fhash in all_file_hashes:
                    orig_path, orig_split = all_file_hashes[fhash]
                    print(f"Warning: Duplicate image content found!")
                    print(f"  - File 1: {src_path} (assigned to {split})")
                    print(f"  - File 2: {orig_path} (assigned to {orig_split})")
                    if split != orig_split:
                        print("  [LEAKAGE DETECTED] Duplicate content assigned to different splits!")
                        leakage_detected = True
                else:
                    all_file_hashes[fhash] = (src_path, split)
                
                # Copy file as-is, untouched
                shutil.copy2(src_path, dst_path)
                
                split_hashes[split].add(fhash)
                split_counts[split][cls] += 1
                
    print("\n--- Split Preparation Summary ---")
    for split in splits:
        total_split = sum(split_counts[split].values())
        print(f"Split '{split}': {total_split} images")
        for cls in classes:
            print(f"  - {cls}: {split_counts[split][cls]}")
            
    # Verify no cross-split duplicate leakage
    print("\n--- Verifying Cross-Split Duplicates ---")
    train_val_intersect = split_hashes["train"].intersection(split_hashes["val"])
    train_test_intersect = split_hashes["train"].intersection(split_hashes["test"])
    val_test_intersect = split_hashes["val"].intersection(split_hashes["test"])
    
    print(f"Duplicate images between Train and Val: {len(train_val_intersect)}")
    print(f"Duplicate images between Train and Test: {len(train_test_intersect)}")
    print(f"Duplicate images between Val and Test: {len(val_test_intersect)}")
    
    if len(train_val_intersect) > 0 or len(train_test_intersect) > 0 or len(val_test_intersect) > 0 or leakage_detected:
        print("\n[CRITICAL ERROR] Cross-split duplicate leakage detected!")
        return False
    else:
        print("\n[SUCCESS] No cross-split duplicate leakage verified!")
        return True

if __name__ == "__main__":
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(workspace_dir, "Watermelon")
    output_dir = os.path.join(workspace_dir, "data", "processed")
    
    # Perform split
    prepare_splits(dataset_dir, output_dir, seed=42)
