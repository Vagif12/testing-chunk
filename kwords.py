import requests
import json

url = "http://0.0.0.0:5000/"

def get_keywords(text):

    payload = json.dumps({
    "text": text,
    "lang": "en",
    "n": 10
    })
    headers = {
    'Content-Type': 'application/json'
    }


    # obtain response
    response = requests.request("POST", url, headers=headers, data=payload)

    keywords, scores, relevant_keywords = response.json()

    # Pair each keyword with its corresponding score
    keyword_score_pairs = list(zip(keywords, scores))

    # Sort the pairs based on the scores (in descending order)
    sorted_keyword_score_pairs = sorted(keyword_score_pairs, key=lambda x: x[1], reverse=True)

    # Extract the sorted keywords and scores separately
    sorted_keywords, sorted_scores = zip(*sorted_keyword_score_pairs)

    return {"keywords":list(sorted_keywords[:8])}

print(get_keywords("Representation, on the other hand, is about ensuring that all stakeholders are included in the decision-making process. This means that the perspectives and needs of all potential users, including those from diverse backgrounds and with diverse needs, are taken into account in the design process. It's about making sure that the product or service is not only usable by a diverse group of people, but also meets their needs and respects their values."))