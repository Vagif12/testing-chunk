import spacy

nlp = spacy.load("en_core_web_md")

import unicodedata,re

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def escape_characters(text):
    words = text.split(' ')
    new_list = []
    for word in words:
        if "\u2019" not in word:
            normalized_text = unicodedata.normalize("NFKD", word)
            stripped_accents = strip_accents(normalized_text)
            cleaned = re.sub(r"[^'\x00-\x7f]", r' ', stripped_accents).strip()
            new_list.append(cleaned)
        else:
            new_list.append(word)

    output = ' '.join(new_list)
    return output.replace('\u2019', "'")


def sliding_window_segmentation(text, window_size=3, overlap=2):
    # Split the text into sentences using a simple regex pattern
    doc = nlp(escape_characters(text))
    sentences = [sent.text for sent in doc.sents]

    # Validate window_size and overlap
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Invalid window_size or overlap parameters.")

    # Initialize lists to store the chunks and the current window
    chunks = []
    window = []

    # Loop through sentences to create chunks
    for i, sentence in enumerate(sentences):
        window.append(sentence)

        # Check if the window size has been reached or it's the last sentence
        if len(window) == window_size or i == len(sentences) - 1:
            chunk = ' '.join(window)
            chunks.append(chunk)

            # Move the window by the specified overlap
            window = window[window_size - overlap:]

    return chunks