# =============================================================================
# 🧬 ReLoop: TACO YOLO Training Notebook
# =============================================================================
# Run this in Google Colab (Runtime → Change runtime type → T4 GPU)
#
# HOW TO USE:
# 1. Open Google Colab: https://colab.research.google.com
# 2. Create a new notebook
# 3. Copy each "CELL" section below into a separate Colab cell
# 4. Run cells in order (Shift+Enter)
# 5. Download the trained model at the end
# =============================================================================


# %% [markdown]
# # 🧬 ReLoop: Train YOLO on TACO (Trash Dataset)
# This notebook fine-tunes YOLOv11m on the TACO dataset with 15 custom waste categories.

# %%
# ===========================================================================
# CELL 1: SETUP & INSTALL DEPENDENCIES
# ===========================================================================
# Check GPU availability
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "⚠️ No GPU detected! Training will be slow.")

# Install required packages
get_ipython().system('pip install -q ultralytics pycocotools Pillow tqdm')

print("✅ Dependencies installed!")


# %%
# ===========================================================================
# CELL 2: DOWNLOAD TACO DATASET
# ===========================================================================
import os
import json
import shutil
from pathlib import Path

# Clone TACO repository (contains annotations + download script)
if not os.path.exists("TACO"):
    get_ipython().system('git clone https://github.com/pedropro/TACO.git')
    print("✅ TACO repository cloned!")
else:
    print("ℹ️ TACO already exists, skipping clone.")

# Install TACO dependencies
get_ipython().system('pip install -q -r TACO/requirements.txt')

# Download images from Flickr (this takes ~10-15 minutes)
os.chdir("TACO")
print("📥 Downloading TACO images from Flickr... This takes ~10-15 minutes.")
get_ipython().system('python download.py')
os.chdir("..")

# Verify download
annotations_path = "TACO/data/annotations.json"
if os.path.exists(annotations_path):
    with open(annotations_path) as f:
        data = json.load(f)
    print(f"✅ TACO loaded: {len(data['images'])} images, {len(data['annotations'])} annotations, {len(data['categories'])} categories")
else:
    raise FileNotFoundError("❌ annotations.json not found! Download may have failed.")


# %%
# ===========================================================================
# CELL 3: EXPLORE TACO CATEGORIES
# ===========================================================================
import json

with open("TACO/data/annotations.json") as f:
    data = json.load(f)

print("=" * 60)
print("TACO DATASET CATEGORIES")
print("=" * 60)

# Count annotations per category
from collections import Counter
cat_counts = Counter()
for ann in data['annotations']:
    cat_counts[ann['category_id']] += 1

cat_id_to_name = {c['id']: c['name'] for c in data['categories']}

for cat_id, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    name = cat_id_to_name.get(cat_id, f"Unknown-{cat_id}")
    print(f"  [{cat_id:3d}] {name:35s} → {count:4d} annotations")

print(f"\nTotal annotations: {sum(cat_counts.values())}")


# %%
# ===========================================================================
# CELL 4: CONVERT TACO → YOLO FORMAT (15 Custom Categories)
# ===========================================================================
import json
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ─── OUR 15 WASTE CATEGORIES ───
CLASS_NAMES = [
    "Plastic Bottle",       # 0
    "Plastic Bag",          # 1
    "Plastic Wrapper",      # 2
    "Plastic Container",    # 3
    "Styrofoam",            # 4
    "Crisp MLP Packet",     # 5  (Multi-Layer Plastic)
    "Glass Bottle",         # 6
    "Broken Glass",         # 7
    "Aluminium Can",        # 8
    "Food Can",             # 9
    "Metal Scrap",          # 10
    "Cardboard",            # 11
    "Paper",                # 12
    "Cigarette",            # 13
    "Hazardous",            # 14
]

# ─── MAP TACO CATEGORY NAMES → OUR 15 CLASS IDs ───
# Keys are lowercase TACO category names → values are our class IDs
TACO_TO_RELOOP = {
    # 0: Plastic Bottle
    "clear plastic bottle": 0,
    "other plastic bottle": 0,

    # 1: Plastic Bag
    "single-use carrier bag": 1,
    "garbage bag": 1,
    "polypropylene bag": 1,
    "plastified paper bag": 1,
    "plastic bag - Loss": 1,

    # 2: Plastic Wrapper
    "other plastic wrapper": 2,
    "plastic film": 2,
    "six pack rings": 2,

    # 3: Plastic Container
    "disposable plastic cup": 3,
    "other plastic cup": 3,
    "other plastic container": 3,
    "spread tub": 3,
    "plastic utensils": 3,
    "plastic lid": 3,
    "plastic bottle cap": 3,
    "plastic straw": 3,
    "plastic gloves": 3,
    "other plastic": 3,
    "foam food container": 3,

    # 4: Styrofoam
    "styrofoam piece": 4,
    "foam cup": 4,

    # 5: Crisp/MLP Packet
    "crisp packet": 5,
    "aluminium blister pack": 5,
    "carded blister pack": 5,
    "squeezable tube": 5,

    # 6: Glass Bottle
    "glass bottle": 6,
    "glass jar": 6,
    "glass cup": 6,

    # 7: Broken Glass
    "broken glass": 7,

    # 8: Aluminium Can
    "drink can": 8,
    "pop tab": 8,

    # 9: Food Can
    "food can": 9,
    "aerosol": 9,

    # 10: Metal Scrap
    "aluminium foil": 10,
    "metal bottle cap": 10,
    "metal lid": 10,
    "scrap metal": 10,

    # 11: Cardboard
    "corrugated carton": 11,
    "pizza box": 11,
    "egg carton": 11,
    "meal carton": 11,
    "other carton": 11,
    "drink carton": 11,

    # 12: Paper
    "normal paper": 12,
    "magazine paper": 12,
    "wrapping paper": 12,
    "paper bag": 12,
    "tissues": 12,
    "toilet tube": 12,
    "paper cup": 12,
    "paper straw": 12,

    # 13: Cigarette
    "cigarette": 13,

    # 14: Hazardous
    "battery": 14,
}

# ─── LOAD ANNOTATIONS ───
with open("TACO/data/annotations.json") as f:
    coco = json.load(f)

cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
img_id_to_info = {img['id']: img for img in coco['images']}

# ─── BUILD PER-IMAGE ANNOTATION LISTS ───
img_annotations = {}
skipped_categories = set()
mapped_count = 0
skipped_count = 0

for ann in coco['annotations']:
    img_id = ann['image_id']
    cat_name = cat_id_to_name.get(ann['category_id'], "")
    cat_name_lower = cat_name.lower().strip()

    # Look up our class ID
    reloop_id = TACO_TO_RELOOP.get(cat_name_lower)
    if reloop_id is None:
        skipped_categories.add(cat_name)
        skipped_count += 1
        continue

    img_info = img_id_to_info.get(img_id)
    if img_info is None:
        continue

    img_w = img_info['width']
    img_h = img_info['height']

    # Convert COCO bbox [x_min, y_min, width, height] → YOLO [x_center, y_center, w, h] (normalized)
    bx, by, bw, bh = ann['bbox']
    x_center = (bx + bw / 2.0) / img_w
    y_center = (by + bh / 2.0) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.001, min(1.0, w_norm))
    h_norm = max(0.001, min(1.0, h_norm))

    if img_id not in img_annotations:
        img_annotations[img_id] = []

    img_annotations[img_id].append(f"{reloop_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    mapped_count += 1

print(f"✅ Mapped {mapped_count} annotations across {len(img_annotations)} images")
print(f"⚠️ Skipped {skipped_count} annotations from unmapped categories: {skipped_categories}")

# ─── CREATE YOLO DIRECTORY STRUCTURE ───
dataset_dir = Path("taco_yolo")
for split in ["train", "val"]:
    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# ─── SPLIT AND COPY FILES ───
image_ids = list(img_annotations.keys())
random.seed(42)
random.shuffle(image_ids)

split_idx = int(len(image_ids) * 0.85)  # 85% train, 15% val
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

copied = {"train": 0, "val": 0}
missing = 0

for img_id, labels in tqdm(img_annotations.items(), desc="Processing images"):
    img_info = img_id_to_info[img_id]
    img_filename = img_info['file_name']  # e.g., "batch_1/000001.jpg"

    # Find the actual image file
    src_path = Path("TACO/data") / img_filename
    if not src_path.exists():
        missing += 1
        continue

    split = "train" if img_id in train_ids else "val"
    safe_name = img_filename.replace("/", "_").replace("\\", "_")

    # Copy image
    dst_img = dataset_dir / "images" / split / safe_name
    shutil.copy2(src_path, dst_img)

    # Write YOLO label file
    label_name = Path(safe_name).stem + ".txt"
    dst_label = dataset_dir / "labels" / split / label_name
    with open(dst_label, "w") as f:
        f.write("\n".join(labels))

    copied[split] += 1

print(f"\n{'=' * 50}")
print(f"📊 DATASET READY:")
print(f"   Train: {copied['train']} images")
print(f"   Val:   {copied['val']} images")
print(f"   Missing images (failed download): {missing}")
print(f"   Classes: {len(CLASS_NAMES)}")
print(f"{'=' * 50}")


# %%
# ===========================================================================
# CELL 5: CREATE DATASET YAML CONFIG
# ===========================================================================
import yaml
from pathlib import Path

dataset_config = {
    'path': str(Path("taco_yolo").resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 15,
    'names': {
        0: 'Plastic Bottle',
        1: 'Plastic Bag',
        2: 'Plastic Wrapper',
        3: 'Plastic Container',
        4: 'Styrofoam',
        5: 'Crisp MLP Packet',
        6: 'Glass Bottle',
        7: 'Broken Glass',
        8: 'Aluminium Can',
        9: 'Food Can',
        10: 'Metal Scrap',
        11: 'Cardboard',
        12: 'Paper',
        13: 'Cigarette',
        14: 'Hazardous',
    }
}

yaml_path = "taco_yolo/taco.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Dataset config written to: {yaml_path}")
print(f"\nContents:")
with open(yaml_path) as f:
    print(f.read())


# %%
# ===========================================================================
# CELL 6: VERIFY DATASET (Visual Sanity Check)
# ===========================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import os

CLASS_NAMES = [
    "Plastic Bottle", "Plastic Bag", "Plastic Wrapper", "Plastic Container",
    "Styrofoam", "Crisp MLP Packet", "Glass Bottle", "Broken Glass",
    "Aluminium Can", "Food Can", "Metal Scrap", "Cardboard",
    "Paper", "Cigarette", "Hazardous"
]

COLORS = [
    '#00FFFF', '#FF6B6B', '#FFA500', '#00FF00', '#FF69B4',
    '#FFD700', '#4169E1', '#FF0000', '#C0C0C0', '#8B4513',
    '#808080', '#D2691E', '#FFFACD', '#FF1493', '#9400D3'
]

train_dir = "taco_yolo/images/train"
label_dir = "taco_yolo/labels/train"

images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
samples = random.sample(images, min(6, len(images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, img_name in enumerate(samples):
    img = Image.open(os.path.join(train_dir, img_name))
    w, h = img.size
    axes[idx].imshow(img)
    axes[idx].set_title(img_name[:30], fontsize=9)
    axes[idx].axis('off')

    # Draw YOLO boxes
    label_file = os.path.join(label_dir, Path(img_name).stem + ".txt")
    if os.path.exists(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                box_w = bw * w
                box_h = bh * h
                color = COLORS[cls_id % len(COLORS)]
                rect = patches.Rectangle((x1, y1), box_w, box_h,
                                         linewidth=2, edgecolor=color, facecolor='none')
                axes[idx].add_patch(rect)
                axes[idx].text(x1, y1-5, CLASS_NAMES[cls_id],
                              color=color, fontsize=8, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

plt.suptitle("TACO Dataset - Sample Images with YOLO Annotations", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("taco_samples.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Verification complete! Check the plot above.")


# %%
# ===========================================================================
# CELL 7: TRAIN YOLOv11m ON TACO (50 Epochs)
# ===========================================================================
from ultralytics import YOLO

# Load YOLOv11m with COCO pretrained weights (transfer learning)
model = YOLO("yolo11m.pt")

print("=" * 60)
print("  🚀 STARTING TRAINING: YOLOv11m on TACO (15 classes)")
print("  📊 Epochs: 50 | Image Size: 640 | Batch: 16")
print("  🎯 Fine-tuning from COCO pretrained weights")
print("=" * 60)

# Train
results = model.train(
    data="taco_yolo/taco.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="reloop_taco_v1",
    patience=20,           # Early stopping: stop if no improvement for 20 epochs
    save=True,
    save_period=25,         # Save checkpoint every 25 epochs
    plots=True,             # Generate training plots
    verbose=True,
    optimizer="AdamW",
    lr0=0.001,              # Initial learning rate
    lrf=0.01,               # Final learning rate factor
    warmup_epochs=5,        # Warmup for 5 epochs
    warmup_momentum=0.8,
    weight_decay=0.0005,
    mosaic=1.0,             # Mosaic augmentation (great for small datasets)
    mixup=0.15,             # MixUp augmentation
    copy_paste=0.1,         # Copy-paste augmentation
    hsv_h=0.015,            # HSV hue augmentation
    hsv_s=0.7,              # HSV saturation augmentation
    hsv_v=0.4,              # HSV value augmentation
    flipud=0.1,             # Vertical flip
    fliplr=0.5,             # Horizontal flip
    degrees=10.0,           # Rotation
    translate=0.2,          # Translation
    scale=0.5,              # Scale
    shear=2.0,              # Shear
    # --- LOSS FUNCTIONS ---
    box=7.5,                # Box CIoU loss gain (Bounding Box Regression)
    cls=0.5,                # Classification BCE loss gain
    dfl=1.5,                # Distribution Focal Loss gain
    fl_gamma=1.5,           # Focal Loss gamma (Addresses TACO dataset class imbalance)
)

print("\n✅ TRAINING COMPLETE!")


# %%
# ===========================================================================
# CELL 8: EVALUATE THE MODEL
# ===========================================================================
from ultralytics import YOLO
import os

# Find the best weights
best_weights = "runs/detect/reloop_taco_v1/weights/best.pt"
if not os.path.exists(best_weights):
    # Search for it
    for root, dirs, files in os.walk("runs"):
        if "best.pt" in files:
            best_weights = os.path.join(root, "best.pt")
            break

print(f"📁 Best weights: {best_weights}")

# Load trained model
model = YOLO(best_weights)

# Validate on the validation set
metrics = model.val(data="taco_yolo/taco.yaml")

print("\n" + "=" * 60)
print("  📊 VALIDATION RESULTS")
print("=" * 60)
print(f"  mAP@50:    {metrics.box.map50:.4f}")
print(f"  mAP@50-95: {metrics.box.map:.4f}")
print(f"  Precision:  {metrics.box.mp:.4f}")
print(f"  Recall:     {metrics.box.mr:.4f}")
print("=" * 60)


# %%
# ===========================================================================
# CELL 9: TEST ON SAMPLE IMAGES
# ===========================================================================
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# Load trained model
best_weights = "runs/detect/reloop_taco_v1/weights/best.pt"
if not os.path.exists(best_weights):
    for root, dirs, files in os.walk("runs"):
        if "best.pt" in files:
            best_weights = os.path.join(root, "best.pt")
            break

model = YOLO(best_weights)

# Run inference on validation images
val_dir = "taco_yolo/images/val"
val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir)
              if f.endswith(('.jpg', '.png', '.jpeg'))]

test_images = random.sample(val_images, min(6, len(val_images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, img_path in enumerate(test_images):
    results = model(img_path, verbose=False)
    annotated = results[0].plot()  # BGR numpy array with boxes drawn

    # Convert BGR to RGB for matplotlib
    annotated_rgb = annotated[:, :, ::-1]
    axes[idx].imshow(annotated_rgb)
    axes[idx].set_title(f"{len(results[0].boxes)} detections", fontsize=11)
    axes[idx].axis('off')

plt.suptitle("TACO-Trained YOLO: Inference on Validation Images", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("taco_inference_results.png", dpi=150, bbox_inches='tight')
plt.show()


# %%
# ===========================================================================
# CELL 10: DOWNLOAD TRAINED MODEL
# ===========================================================================
import shutil
from google.colab import files

# Find best weights
best_weights = "runs/detect/reloop_taco_v1/weights/best.pt"
if not os.path.exists(best_weights):
    for root, dirs, files_list in os.walk("runs"):
        if "best.pt" in files_list:
            best_weights = os.path.join(root, "best.pt")
            break

# Copy with a clear name
final_name = "reloop_taco_yolo.pt"
shutil.copy2(best_weights, final_name)

# Get file size
size_mb = os.path.getsize(final_name) / (1024 * 1024)

print("=" * 60)
print(f"  ✅ MODEL READY: {final_name} ({size_mb:.1f} MB)")
print("=" * 60)
print(f"\n  📥 Downloading to your computer...")
print(f"  After download, place this file in your project root:")
print(f"  residuav2-main/reloop_taco_yolo.pt")
print(f"\n  Then update urban_miner.py line 55:")
print(f'  object_model = YOLO("reloop_taco_yolo.pt")')

# Trigger download in Colab
files.download(final_name)

# Also download training plots
plots_dir = "runs/detect/reloop_taco_v1"
if os.path.exists(plots_dir):
    shutil.make_archive("training_plots", 'zip', plots_dir)
    print("\n  📊 Also downloading training plots...")
    files.download("training_plots.zip")


# %%
# ===========================================================================
# CELL 11: PRINT CLASS MAPPING (for urban_miner.py update)
# ===========================================================================

print("""
# ══════════════════════════════════════════════════════════════
#  COPY THIS INTO urban_miner.py (replace existing class maps)
# ══════════════════════════════════════════════════════════════

# TACO-trained YOLO class names → Material categories
TACO_CLASS_TO_MATERIAL = {
    "Plastic Bottle":     "Plastic",
    "Plastic Bag":        "Plastic",
    "Plastic Wrapper":    "Plastic",
    "Plastic Container":  "Plastic",
    "Styrofoam":          "Plastic",
    "Crisp MLP Packet":   "MLP",
    "Glass Bottle":       "Glass",
    "Broken Glass":       "Glass",
    "Aluminium Can":      "Metal",
    "Food Can":           "Metal",
    "Metal Scrap":        "Metal",
    "Cardboard":          "Paper",
    "Paper":              "Paper",
    "Cigarette":          "Hazardous",
    "Hazardous":          "Hazardous",
}
""")
