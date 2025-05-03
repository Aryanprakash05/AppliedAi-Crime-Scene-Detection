from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small")

crime_label_map = {
    "knife": "Weapon",
    "handbag": "Victim’s Bag",
    "person": "Victim/Suspect",
    "wine glass": "Drink (Potential Evidence)",
    "cell phone": "Broken Phone",
    "couch": "Struggle Area",
    "remote": "Object Nearby",
    "scissors": "Sharp Object",
    "bottle": "Glass Object",
    "chair": "Furniture",
    "sink": "Possible Cleaning",
    "cup": "Glassware",
    "book": "Documents",
    "laptop": "Digital Evidence",
    "tv": "Surveillance Angle",
    "potted plant": "Object (Needs Investigation for prints)" , 
    "car" : "possibe eye witness"
}


def detect_objects(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        coco_label = model.config.id2label[label.item()]
        crime_label = crime_label_map.get(coco_label, f"{coco_label}")
        draw_box(ax, box, crime_label, score.item())

    plt.axis("off")
    plt.title("🔍 Crime Scene Scan — Detected Objects")
    plt.show()

def draw_box(ax, box, label, score):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='crimson', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin - 5, f"{label} ({score:.2f})", color="white", fontsize=9,
            bbox=dict(facecolor="crimson", edgecolor="none", boxstyle="round,pad=0.3"))

if __name__ == "__main__":
    test_image = "data/images/crimescene8.jpg"
    if os.path.exists(test_image):
        detect_objects(test_image)
    else:
        print(f"⚠️ Image not found: {test_image}")
