# src/prepare_efficiency_dataset.py - COMPLETE FIXED VERSION
"""
Prepare dataset with both defect classification and efficiency regression labels
COMPLETE VERSION - Handles everything from start to finish
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import random


def efficiency_range_to_numeric(range_str):
    """
    Convert efficiency range (e.g., "70-80%") to numeric value (midpoint)
    """
    if pd.isna(range_str):
        return None

    # Extract numbers from range like "70-80%"
    import re
    numbers = re.findall(r'\d+', str(range_str))

    if len(numbers) >= 2:
        low = int(numbers[0])
        high = int(numbers[1])
        return (low + high) / 2.0
    elif len(numbers) == 1:
        return float(numbers[0])
    else:
        return None


def create_efficiency_dataset():
    """
    Create enhanced dataset with efficiency labels
    """

    # Paths
    base_dir = r"C:\Users\Adm\Desktop\Kalpana\python projects\solar_defects_classification"
    excel_path = os.path.join(base_dir, "efficiency_report.xlsx")
    base_data_path = os.path.join(base_dir, "data", "solar_defects_dataset")
    enhanced_data_path = os.path.join(base_dir, "data", "enhanced_solar_dataset")

    print(f"📁 Base directory: {base_dir}")
    print(f"📊 Excel file: {excel_path}")
    print(f"📂 Source images: {base_data_path}")
    print(f"📂 Target dataset: {enhanced_data_path}")

    # Check if Excel file exists
    if not os.path.exists(excel_path):
        print(f"❌ Error: Excel file not found at {excel_path}")
        return None

    # Check if source dataset exists
    if not os.path.exists(base_data_path):
        print(f"❌ Error: Source dataset not found at {base_data_path}")
        return None

    # Load efficiency data
    df = pd.read_excel(excel_path)
    print(f"📊 Loaded {len(df)} efficiency records")

    # Convert efficiency ranges to numeric values
    df['efficiency_numeric'] = df['Efficiency Range'].apply(efficiency_range_to_numeric)

    # Remove rows with invalid efficiency data
    df = df.dropna(subset=['efficiency_numeric'])
    print(f"📊 {len(df)} records with valid efficiency data")

    # Print statistics
    print("\n📈 Efficiency Statistics by Category:")
    for category in df['Category'].unique():
        cat_data = df[df['Category'] == category]
        avg_eff = cat_data['efficiency_numeric'].mean()
        count = len(cat_data)
        print(f"  {category}: {avg_eff:.1f}% average ({count} images)")

    # Create enhanced dataset directory
    if os.path.exists(enhanced_data_path):
        print(f"🗑️ Removing existing enhanced dataset...")
        shutil.rmtree(enhanced_data_path)

    os.makedirs(enhanced_data_path, exist_ok=True)

    # Create category directories
    categories = df['Category'].unique()
    for category in categories:
        os.makedirs(os.path.join(enhanced_data_path, category), exist_ok=True)

    # Copy images and create efficiency mapping
    efficiency_mapping = {}
    copied_count = 0
    missing_count = 0

    print(f"\n🔄 Processing images...")

    for index, row in df.iterrows():
        category = row['Category']
        image_name = row['Image']
        efficiency = row['efficiency_numeric']

        # Construct source path
        source_path = os.path.join(base_data_path, category, image_name)

        # Check if source file exists
        if os.path.exists(source_path):
            # Copy to enhanced dataset
            dest_path = os.path.join(enhanced_data_path, category, image_name)

            try:
                shutil.copy2(source_path, dest_path)

                # Store efficiency mapping (use forward slashes for consistency)
                relative_path = f"{category}/{image_name}"
                efficiency_mapping[relative_path] = efficiency

                copied_count += 1

                # Progress indicator
                if copied_count % 50 == 0:
                    print(f"   ✅ Processed {copied_count}/{len(df)} images...")

            except Exception as e:
                print(f"   ❌ Error copying {source_path}: {str(e)}")
                missing_count += 1
        else:
            # Try different case variations
            category_path = os.path.join(base_data_path, category)
            if os.path.exists(category_path):
                # List actual files in the category folder
                actual_files = os.listdir(category_path)

                # Try case-insensitive match
                found_match = False
                for actual_file in actual_files:
                    if actual_file.lower() == image_name.lower():
                        # Found a match with different case
                        actual_source = os.path.join(category_path, actual_file)
                        dest_path = os.path.join(enhanced_data_path, category, actual_file)

                        try:
                            shutil.copy2(actual_source, dest_path)
                            relative_path = f"{category}/{actual_file}"
                            efficiency_mapping[relative_path] = efficiency
                            copied_count += 1
                            found_match = True
                            break
                        except Exception as e:
                            print(f"   ❌ Error copying {actual_source}: {str(e)}")

                if not found_match:
                    print(f"   ⚠️ Missing: {category}/{image_name}")
                    missing_count += 1
            else:
                print(f"   ❌ Category folder not found: {category_path}")
                missing_count += 1

    print(f"\n✅ Dataset preparation complete:")
    print(f"   📁 Successfully copied: {copied_count} images")
    print(f"   ❌ Missing/failed: {missing_count} images")
    print(f"   📍 Enhanced dataset location: {enhanced_data_path}")

    if copied_count == 0:
        print(f"\n❌ No images were copied!")
        return None

    # Save efficiency mapping
    efficiency_df = pd.DataFrame([
        {'image_path': path, 'efficiency': eff}
        for path, eff in efficiency_mapping.items()
    ])

    mapping_path = os.path.join(enhanced_data_path, 'efficiency_mapping.csv')
    efficiency_df.to_csv(mapping_path, index=False)
    print(f"   💾 Efficiency mapping saved: {mapping_path}")

    return enhanced_data_path, efficiency_mapping


def split_dataset(enhanced_data_path, efficiency_mapping, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train/validation/test sets while maintaining efficiency labels
    """
    print(f"\n📂 Splitting dataset...")

    # Set random seed for reproducible splits
    random.seed(42)

    # Convert mapping to DataFrame
    efficiency_df = pd.DataFrame([
        {'image_path': path, 'efficiency': eff}
        for path, eff in efficiency_mapping.items()
    ])

    # Group by category for stratified split
    splits = {'train': [], 'val': [], 'test': []}

    # Get unique categories
    categories = efficiency_df['image_path'].apply(lambda x: x.split('/')[0]).unique()
    print(f"📂 Categories: {list(categories)}")

    for category in categories:
        # Get all images for this category
        category_data = efficiency_df[efficiency_df['image_path'].str.startswith(category + '/')]
        category_list = category_data.to_dict('records')

        # Shuffle the data
        random.shuffle(category_list)

        # Calculate split indices
        n_total = len(category_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split the data
        train_data = category_list[:n_train]
        val_data = category_list[n_train:n_train + n_val]
        test_data = category_list[n_train + n_val:]

        print(f"   📁 {category}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # Add to splits
        splits['train'].extend(train_data)
        splits['val'].extend(val_data)
        splits['test'].extend(test_data)

    # Save split files
    print(f"\n💾 Saving split files...")

    total_saved = 0
    for split_name, split_data in splits.items():
        if split_data:  # Only save if there's data
            split_df = pd.DataFrame(split_data)
            split_path = os.path.join(enhanced_data_path, f'{split_name}_split.csv')
            split_df.to_csv(split_path, index=False)
            total_saved += len(split_data)
            print(f"   ✅ {split_name}_split.csv: {len(split_data)} samples")
        else:
            print(f"   ❌ No data for {split_name} split")

    # Verify files were created
    print(f"\n🔍 Verifying split files:")
    required_files = ['train_split.csv', 'val_split.csv', 'test_split.csv']

    all_created = True
    for file in required_files:
        file_path = os.path.join(enhanced_data_path, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file}: {len(df)} records")

                # Verify the file has correct structure
                if 'image_path' not in df.columns or 'efficiency' not in df.columns:
                    print(f"      ⚠️ Warning: Missing expected columns")
                    all_created = False

            except Exception as e:
                print(f"   ❌ {file}: Error reading - {str(e)}")
                all_created = False
        else:
            print(f"   ❌ {file}: Not found")
            all_created = False

    if all_created:
        print(f"\n📊 Final dataset splits:")
        print(f"   🏋️ Training: {len(splits['train'])} images ({len(splits['train']) / total_saved * 100:.1f}%)")
        print(f"   ✅ Validation: {len(splits['val'])} images ({len(splits['val']) / total_saved * 100:.1f}%)")
        print(f"   🔬 Test: {len(splits['test'])} images ({len(splits['test']) / total_saved * 100:.1f}%)")
        print(f"   📊 Total: {total_saved} images")

    return splits if all_created else None


def verify_dataset(enhanced_data_path):
    """
    Final verification of the complete dataset
    """
    print(f"\n🔍 Final dataset verification...")

    required_files = [
        'efficiency_mapping.csv',
        'train_split.csv',
        'val_split.csv',
        'test_split.csv'
    ]

    all_good = True

    for file in required_files:
        file_path = os.path.join(enhanced_data_path, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file}: {len(df)} records")

                # Quick data sample
                if len(df) > 0:
                    sample = df.head(1).to_dict('records')[0]
                    print(f"      📋 Sample: {sample}")

            except Exception as e:
                print(f"   ❌ {file}: Error - {str(e)}")
                all_good = False
        else:
            print(f"   ❌ {file}: Missing")
            all_good = False

    # Check if category folders have images
    categories = ['bird_droppings', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']

    print(f"\n📂 Category folders:")
    for category in categories:
        category_path = os.path.join(enhanced_data_path, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   📁 {category}: {len(files)} images")
            if len(files) == 0:
                all_good = False
        else:
            print(f"   ❌ {category}: Folder missing")
            all_good = False

    return all_good


if __name__ == "__main__":
    print("🚀 Creating Enhanced Solar Panel Dataset with Efficiency Labels")
    print("=" * 70)

    # Step 1: Create enhanced dataset and copy images
    result = create_efficiency_dataset()

    if not result:
        print(f"\n❌ Dataset creation failed. Exiting.")
        exit(1)

    enhanced_path, mapping = result

    # Step 2: Split dataset into train/val/test
    splits = split_dataset(enhanced_path, mapping)

    if not splits:
        print(f"\n❌ Dataset splitting failed. Exiting.")
        exit(1)

    # Step 3: Verify everything is correct
    verification_passed = verify_dataset(enhanced_path)

    if verification_passed:
        print(f"\n🎉 SUCCESS! Enhanced dataset ready for AI training!")
        print(f"   📁 Dataset location: {enhanced_path}")
        print(f"   🏷️ Labels: Defect type + Efficiency percentage")
        print(f"   📊 Total images: {len(mapping)}")
        print(f"   🤖 Ready to run: python src/train_multi_output.py")
    else:
        print(f"\n⚠️ Dataset created but verification found issues.")
        print(f"   Check the error messages above before proceeding.")