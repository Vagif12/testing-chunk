from transformers import pipeline, AutoTokenizer

# Load the local model and set up the pipeline
model_name = "Gladiator/microsoft-deberta-v3-large_ner_conll2003"
pipe = pipeline("token-classification", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def ner(text):
    results = pipe(text)

    # Convert IOB2 format to the custom format of the API response
    entities = []
    current_entity = None
    for token_info in results:
        if token_info["entity"].startswith("B-"):
            # Start of a new entity
            current_entity = {
                "entity_group": token_info["entity"][2:],  # Remove the "B-" prefix
                "word": token_info["word"],
                "start": token_info["start"],
                "end": token_info["end"],
            }
            entities.append(current_entity)
        elif token_info["entity"].startswith("I-") and current_entity:
            # Inside an entity, append the word to the current entity
            current_entity["word"] += token_info["word"]
            current_entity["end"] = token_info["end"]
        else:
            # Outside an entity or non-entity token, reset the current entity
            current_entity = None

    # Decode word pieces back to original tokens
    for entity in entities:
        word_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(entity["word"], add_special_tokens=False))
        entity["word"] = tokenizer.convert_tokens_to_string(word_tokens)

    # Remove the "score" key from the results
    for entity in entities:
        entity.pop("score", None)

    return {"entities": entities}

# Example usage
txt = "Vagif Aliyev is the founder of Snapstudy.ai"
result = ner(txt)
print(result)
