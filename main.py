from extract_nlp import extract_entities
from summarize import summarize_text
from graph_builder import extract_graph
from detect import detect_objects
import os

try:
    from gan_generator import CrimeSceneGenerator
    GEN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Scene generator not available: {str(e)}")
    GEN_AVAILABLE = False

def run_pipeline():
    report_path = "data/reports/sample_report.txt"
    image_path = "data/images/crimescene8.jpg"

    if not os.path.exists(report_path):
        print(f"❌ Report not found at {report_path}")
        return

    with open(report_path, "r") as f:
        report = f.read()

    print("\n📝 Original Report:\n")
    print(report)

    print("\n📌 Named Entities:")
    extract_entities(report)

    print("\n🧠 Summary:")
    summary = summarize_text(report)
    print(summary)

    print("\n📊 Relationship Graph:")
    relations = extract_graph(report)
    for source, rel, target in relations:
        print(f"[{source}] --{rel}--> [{target}]")

    print("\n🧭 Object Detection (Crime Scene Image):")
    if os.path.exists(image_path):
        detect_objects(image_path)
    else:
        print(f"⚠️ Image not found: {image_path}")

    # if GEN_AVAILABLE:
    #     print("\n🎨 Generating Crime Scene Visualization from Report...")
    #     try:
    #         generator = CrimeSceneGenerator()
    #         generated_path = generator.generate_scene(report)
    #         print(f"✅ Generated crime scene image saved to: {generated_path}")
    #     except Exception as e:
    #         print(f"❌ Failed to generate crime scene: {str(e)}")
    # else:
    #     print("\n⚠️ Scene generator not available")

if __name__ == "__main__":
    run_pipeline()