#!/usr/bin/env python3

# % cat testfiles-5/predictions.tsv | sort | python3 explainer.py
# In paragraph 4, sentence 2, the participant "plants" is moved from an unknown location to sediment
# In paragraph 4, sentence 3, the participant "bacteria" is moved from an unknown location to sediment
# In paragraph 4, sentence 8, the participant "plants" is moved from sediment to one mile underground
# In paragraph 4, sentence 8, the participant "sediment" is moved from an unknown location to underground
# In paragraph 4, sentence 10, the participant "bacteria" is destroyed at "sediment"
# In paragraph 4, sentence 10, the participant "oil" is created at "underground"
# In paragraph 4, sentence 10, the participant "plants" is destroyed at "one mile underground"


import sys

explanations = []
for line in sys.stdin:
    line = line.strip()
    paragraph_id, sentence, participant, action, location_before, location_after = line.split("\t")

    event = ""
    if action == "DESTROY":
        if location_before == "?":
            location_before = f"an unknown location"
        event = f"destroyed at `{location_before}`"
    elif action == "CREATE":
        if location_after == "?":
            location_after = f"an unknown location"
        event = f"created at `{location_after}`"
    elif action == "MOVE":
        if location_before == "?":
            location_before = f"an unknown location"
        if location_after == "?":
            location_after = f"an unknown location"
        event = f"moved from `{location_before}` to `{location_after}`"

    if event:
        explanation = f"In paragraph {paragraph_id}, sentence {sentence}, the participant `{participant}` is {event}"
        explanations.append((int(paragraph_id), int(sentence), explanation))

for _, _, explanation in sorted(explanations):
    print(explanation)
