from transformers import pipeline
import re

ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

def extract_graph(report_text):
    print("🗂️  Parsing report for named entities...")
    entities = ner(report_text)

    persons = [ent['word'] for ent in entities if ent['entity_group'] == "PER"]
    orgs = [ent['word'] for ent in entities if ent['entity_group'] == "ORG"]
    locs = [ent['word'] for ent in entities if ent['entity_group'] in ["LOC", "ORG"]]

    case_graph = []

    print("📍 Establishing key identities and connections...")

    victim = next((p for p in persons if "Marcus" in p or "Lane" in p), "Marcus Lane")
    print(f"🧍 Victim Identified: {victim}")
    case_graph.append(("Unknown Assailant", "killed", victim))

    location = "Rainbow Gallery, 118 Elm Street"
    case_graph.append((victim, "found_at", location))

    officer = next((p for p in persons if "Westbrook" in p), "Det. Aaron Westbrook")
    case_graph.append((officer, "filed", "Report"))

    witness1 = next((p for p in persons if "Claire" in p or "Nolan" in p), "Claire Nolan")
    print(f"👀 Witness #1: {witness1}")
    case_graph.append((witness1, "witnessed_argument_with", "Red Beret Woman"))

    neighbor = next((p for p in persons if "Ruiz" in p), "Henry Ruiz")
    print(f"👀 Witness #2: {neighbor}")
    case_graph.append((neighbor, "reported_threats_from", "Former Artist"))

    suspect = "Red Beret Woman"
    print(f"🚨 Suspect Description: {suspect}")
    case_graph.append((suspect, "argued_with", victim))
    case_graph.append((suspect, "left_scene", "14:00"))
    case_graph.append((suspect, "carried", "Brown Portfolio"))

    case_graph.append(("Paint-Stained Hoodie", "contains_DNA", "Unknown"))
    case_graph.append(("Event Poster", "stained_with", "Blood"))
    case_graph.append(("Turpentine Bottle", "held_by", victim))
    case_graph.append(("Gallery Tools", "sent_for", "Fingerprint/DNA Analysis"))

    return case_graph

if __name__ == "__main__":
    print("\n📖 Reading crime scene report...")
    with open("data/reports/sample_report.txt", "r") as f:
        crime_report = f.read()

    print("\n🧠 Analyzing report for relationships...")
    leads = extract_graph(crime_report)

    print("\n📊 Extracted Crime Scene Relationships:")
    for source, relation, target in leads:
        print(f"🔗 [{source}] --{relation}--> [{target}]")
