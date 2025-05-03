
from transformers import pipeline

ner_pipeline = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

def extract_entities(text):
    entities = ner_pipeline(text)
    for entity in entities:
        print(f"{entity['word']} -> {entity['entity_group']} ({entity['score']:.2f})")
    return entities
