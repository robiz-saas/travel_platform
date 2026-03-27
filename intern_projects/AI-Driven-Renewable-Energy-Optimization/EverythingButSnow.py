import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def process_folder(folder_path, output_excel="efficiency_report.xlsx"):
    results = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            filepath = os.path.join(root, filename)

            # Extract relative path
            relative_path = os.path.relpath(filepath, folder_path)
            parts = relative_path.split(os.sep)

            # Category = first subfolder (if present)
            category = parts[0] if len(parts) > 1 else "Root"
            image_name = parts[-1]

            try:
                eff = estimate_efficiency(filepath, show=False)
                results.append({
                    "Category": category,
                    "Image": image_name,
                    "Efficiency Range": eff
                })
            except Exception as e:
                print(f"❌ Error with {relative_path}: {e}")
                results.append({
                    "Category": category,
                    "Image": image_name,
                    "Efficiency Range": "Error"
                })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Excel saved to: {output_excel}")

def show_pixel_classification(img):
    output = np.zeros_like(img)
    h, w = img.shape[:2]

    for y in range(h):
        for x in range(w):
            pixel = img[y, x]
            if is_white(pixel):
                output[y, x] = [255, 255, 255]  # White = Ignored
            elif is_blue(pixel):
                output[y, x] = [0, 255, 0]      # Green = Clean
            else:
                output[y, x] = [0, 0, 255]      # Red = Dirty

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Pixel Classification (Green=Clean, Red=Dirty, White=Ignored)")
    plt.axis('off')
    plt.show()

def is_blue(pixel):
    b, g, r = map(int, pixel)
    return (
        b > 50 and
        b >= r and b >= g and
        (b - r) > 5 and (b - g) > 5 and
        not is_white(pixel) and
        (b + g + r) < 700  # safe sum without overflow
    )





def is_white(pixel, threshold=210):
    return np.all(pixel > threshold)

def crop_central_region(img, margin_ratio=0.1):
    h, w = img.shape[:2]
    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)
    return img[margin_h:h - margin_h, margin_w:w - margin_w]

def estimate_efficiency(image_path, show=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    img = cv2.resize(img, (800, 800))

    cropped = crop_central_region(img)

    h, w = cropped.shape[:2]
    clean_pixels = 0
    total_pixels = 0

    for y in range(h):
        for x in range(w):
            pixel = cropped[y, x]
            if is_white(pixel):
                continue
            total_pixels += 1
            if is_blue(pixel):
                clean_pixels += 1

    if total_pixels == 0:
        efficiency = 0
    else:
        efficiency = (clean_pixels / total_pixels) * 100

    efficiency_range = f"{int(efficiency // 10) * 10}-{min(int(efficiency // 10) * 10 + 10, 100)}%"
    print(f"🔋 Estimated Power Efficiency: {efficiency_range}")

    if show:
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title("Central Cropped Region")
        plt.axis('off')
        plt.show()

        show_pixel_classification(cropped)

    return efficiency_range

  
# === USAGE ===
# process_folder(r"C:\Users\Vishisht\Desktop\Mis archivos\AI Solar panel\data\solar_defects_dataset")
estimate_efficiency("Dust (35).jpg")

