# ----------------------------------------
# EXEMPLE 3 - CREDIT SCORING NARRATIF (FINAL)
# ----------------------------------------

import pandas as pd
import re
import json
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------
# 1. DONNEES (soft information)
# ----------------------------------------

data = {
    "text": [
        "The company has a clear strategy and strong reputation with consistent execution",
        "The project lacks clear direction and has a weak reputation",
        "Management presents a structured plan with coherent vision",
        "The strategy is inconsistent and the project is confusing",
        "The firm has a strong brand and a well defined strategy",
        "There are concerns about credibility and unclear positioning"
    ],
    "target": [1, 0, 1, 0, 1, 0]  # 1 = bon crédit, 0 = mauvais
}

df = pd.DataFrame(data)

# ----------------------------------------
# 2. PREPARATION (regex + nettoyage)
# ----------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ----------------------------------------
# 3. LLM SIMULE (soft information)
# ----------------------------------------

def credit_llm_api(text):
    """
    Simulation LLM basé sur stratégie / réputation / cohérence
    """

    # mots clés
    strategie_pos = ["strategy", "clear", "structured", "defined"]
    strategie_neg = ["unclear", "inconsistent"]

    reputation_pos = ["strong", "good", "brand"]
    reputation_neg = ["weak", "concerns"]

    coherence_pos = ["consistent", "coherent"]
    coherence_neg = ["confusing"]

    strategie = 0
    reputation = 0
    coherence = 0

    for word in strategie_pos:
        if word in text:
            strategie += 1
    for word in strategie_neg:
        if word in text:
            strategie -= 1

    for word in reputation_pos:
        if word in text:
            reputation += 1
    for word in reputation_neg:
        if word in text:
            reputation -= 1

    for word in coherence_pos:
        if word in text:
            coherence += 1
    for word in coherence_neg:
        if word in text:
            coherence -= 1

    response = {
        "strategie_score": strategie,
        "reputation_score": reputation,
        "coherence_score": coherence
    }

    return json.dumps(response)

# ----------------------------------------
# 4. JSON → FEATURES
# ----------------------------------------

def json_to_features(json_str):
    data = json.loads(json_str)

    strategie = data["strategie_score"]
    reputation = data["reputation_score"]
    coherence = data["coherence_score"]

    return pd.Series([strategie, reputation, coherence])

df["llm_output"] = df["clean_text"].apply(credit_llm_api)

df[["strategie", "reputation", "coherence"]] = df["llm_output"].apply(json_to_features)

print("\n--- DATA CREDIT SCORING ---")
print(df)

# ----------------------------------------
# 5. MACHINE LEARNING
# ----------------------------------------

X = df[["strategie", "reputation", "coherence"]]
y = df["target"]

model = RandomForestClassifier()
model.fit(X, y)

# ----------------------------------------
# 6. TEST
# ----------------------------------------

test_text = "The company has a clear strategy but weak reputation and confusing execution"

clean = clean_text(test_text)
llm_json = credit_llm_api(clean)

features = json_to_features(llm_json).to_frame().T
features.columns = ["strategie", "reputation", "coherence"]

prediction = model.predict(features)[0]

print("\n--- TEST ---")
print("Texte :", test_text)
print("LLM JSON :", llm_json)
print("Features :", features.values)
print("Prediction (1=bon crédit / 0=mauvais) :", prediction)
