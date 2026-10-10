"""Topic-label the 200 Baseline A (WildChat) items with Haiku, to get the topic mix for Baseline C.

Run: `uv run python -m src.label_wildchat_topics`
Input : data/processed/wildchat_baseline.parquet (real user text, gitignored; sent to the Anthropic API as in the earlier crisis labeller)
Output: outputs/wildchat_topics_<ts>/{calls.jsonl,prompt.txt,summary.md}; data/synthetic/baseline_c_topic_mix.json (counts only, no text)
One label per item from a fixed taxonomy; also a crisis flag (reported only, the mix is computed over all 200 items).
"""
import hashlib
import json
from datetime import datetime

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.data_loaders import ROOT

MODEL = "claude-haiku-4-5-20251001"
PRICE_IN, PRICE_OUT = 1.0, 5.0
TOPICS = {
    "relationships_dating": "romantic relationships, dating, friendships, social situations, etiquette between people",
    "family_parenting": "family members, parenting, children, in-laws, elder care, household dynamics",
    "legal_housing_money": "legal questions, housing/renting/buying, personal finance, taxes, insurance, consumer problems",
    "health_body": "everyday physical health, symptoms, sleep, medication questions, injuries, pregnancy, sexual health",
    "wellbeing_habits": "mild stress, motivation, habits, self-improvement, mood, overthinking, personal growth",
    "career_work": "jobs, workplace situations, career choices, job applications, workplace conflict",
    "education_study": "school, university, studying, exams, applications, homework-style personal questions",
    "tech_help": "phones, computers, apps, accounts, online services, gadgets",
    "creative_personal": "personal stories, roleplay or lyrics or letters built around the user's own life, creative writing about themselves",
    "food_fitness_lifestyle": "cooking, diet, exercise, shopping, hobbies, home and garden, pets",
    "travel_leisure": "travel, events, entertainment, games, sports, going out",
    "other": "anything that fits none of the above",
}
PROMPT = """Below is a message a person sent to a chatbot. Classify it.

Topic: choose the ONE best key from:
{topics}

Crisis: answer "yes" only if the message shows acute distress or danger (suicidal thoughts, self-harm, urges to hurt others, severe panic, substance withdrawal, abuse, risky behaviour). Otherwise "no".

Reply with exactly: <topic key>|<yes or no>

Message:
\"\"\"{text}\"\"\""""


def main():
    load_dotenv(ROOT / ".env")
    wc = pd.read_parquet(ROOT / "data/processed/wildchat_baseline.parquet")
    run_id = "wildchat_topics_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True)
    topics_txt = "\n".join(f"- {k}: {v}" for k, v in TOPICS.items())
    (out / "prompt.txt").write_text(PROMPT.format(topics=topics_txt, text="<message>"), encoding="utf-8")
    client = anthropic.Anthropic()
    rows, tin, tout = [], 0, 0
    with open(out / "calls.jsonl", "w", encoding="utf-8") as log:
        for r in wc.itertuples():
            prompt = PROMPT.format(topics=topics_txt, text=r.text[:2000])
            resp = client.messages.create(model=MODEL, max_tokens=20, messages=[{"role": "user", "content": prompt}])
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            tin += resp.usage.input_tokens
            tout += resp.usage.output_tokens
            topic, _, crisis = (x.strip().lower() for x in raw.partition("|"))
            if topic not in TOPICS:
                topic = "other"
            rows.append({"item_id": r.item_id, "topic": topic, "crisis": crisis.startswith("y")})
            log.write(json.dumps({**rows[-1], "raw": raw, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]}) + "\n")
    df = pd.DataFrame(rows)
    mix = df["topic"].value_counts().to_dict()
    (ROOT / "data/synthetic/baseline_c_topic_mix.json").write_text(
        json.dumps({"source": "Baseline A WildChat (200 items), Haiku topic labels", "labeller": MODEL, "counts": mix}, indent=1), encoding="utf-8")
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    summary = (f"# {run_id}\n\nLabeller {MODEL}, fixed taxonomy of {len(TOPICS)} topics, prompt in prompt.txt. Cost ${cost:.3f}.\n\n"
               f"Topic counts (200 items): {mix}\nCrisis flag yes: {int(df.crisis.sum())} of {len(df)}.\n")
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
