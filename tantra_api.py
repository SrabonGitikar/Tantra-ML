# ═══════════════════════════════════════════════════════════════════════════
# Tantric Deity Semantic Expansion Pipeline — Google AI Studio Edition
# Model: Gemma 4 via Google AI Studio
# ═══════════════════════════════════════════════════════════════════════════

import os, json, re, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# ── API Key ──────────────────────────────────────────────────────────────────
try:
    user_secrets = UserSecretsClient()
    GEMMA4_API_KEY = user_secrets.get_secret("GEMMA4_API_KEY")
    genai.configure(api_key=GEMMA4_API_KEY)
    print("✓ Gemma 4 API key loaded from Kaggle Secrets.")
except Exception as e:
    GEMMA4_API_KEY = os.environ.get("GEMMA4_API_KEY", "")
    if GEMMA4_API_KEY:
        genai.configure(api_key=GEMMA4_API_KEY)
        print("✓ GEMMA4_API_KEY loaded from environment.")
    else:
        raise EnvironmentError(
            f"Could not load GEMMA4_API_KEY — {e}\n"
            "Add it via Kaggle Add-ons → Secrets with the key name 'GEMMA4_API_KEY'."
        )

# ── Model setup ───────────────────────────────────────────────────────────────
# Verify the exact model ID at: aistudio.google.com/models
# Common candidates: "gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"
MODEL_ID   = "gemini-3.1-flash-lite"
CACHE_PATH = "/kaggle/working/domain_expansions_cache.json"

# ── Prompts ───────────────────────────────────────────────────────────────────
EXPANSION_SYSTEM_PROMPT = """
You are an expert in Sanskrit, Hindu Tantra, Shakta Agamas, Vajrayana Buddhism,
and comparative esoteric theology.

Your task: Given a short deity domain description, return a comma-separated list
of 15–25 NOUNS AND NOUN PHRASES that represent its complete esoteric semantic field.

STRICT RULES:
1. Output ONLY nouns and noun phrases — no verbs, no adjectives, no sentences.
2. Always include the direct Sanskrit equivalent(s) first.
3. Include theologically adjacent concepts that share the same ritual/cosmological
   function, even if the surface words differ (e.g., "sound" and "speech" and "mantra"
   all belong to the same field of vak-shakti).
4. DO NOT include form-descriptors (fierce, wrathful, dark, primordial, divine) —
   these describe the deity's appearance, not the governance domain.
5. Aim for specificity over generality: "vak-shakti" is better than "power".
6. Return ONLY the comma-separated list. No preamble, no explanation, no JSON.

EXAMPLES:

Input: "Fierce Speech Power"
Output: vak-shakti, vak, speech, divine speech, mantra, mantra-shakti, nada, primordial sound, shabda, shabda-brahman, knowledge transmission, upadesa, sarasvata-vidya, word, logos, sonic vibration, sacred utterance, tongue, eloquence, poetry, arts, kala

Input: "Primordial Sound"
Output: nada, nada-brahman, shabda, shabda-brahman, primordial vibration, vak, vak-shakti, speech, divine speech, mantra, mantra-power, sonic cosmos, OM, pranava, anahata, unstruck sound, resonance, sarasvata-vidya, logos, knowledge, transmission of teaching

Input: "Time and Dissolution"
Output: kala, mahakala, kalachakra, pralaya, cosmic dissolution, mrityu, death, mahapralaya, destruction of worlds, end of cycle, tamas, darkness as cosmic principle, void, shunya, annihilation, entropy, doomsday, cremation ground, shmashanika, nirvana as dissolution

Input: "Knowledge and Speech"
Output: jnana, vak, vak-shakti, speech, divine knowledge, omniscience, sarvajna, sarasvata-vidya, Vedas, shastra, arts, kala, mantra, shabda, logos, learning, wisdom, eloquence, sarasvata, transmission of teaching, upadesa, sacred texts

Input: "Self-Decapitation and Blood Offering"
Output: shirash-cheda, decapitation, ego-death, atma-samarpana, self-offering, bali, blood offering, sacrifice, chod, egolessness, anatta, sunyata via ego-dissolution, severed head, tapas, self-immolation, karma-kshaya, renunciation
"""

# ── Model init + availability check ──────────────────────────────────────────
gemini_model = genai.GenerativeModel(
    model_name=MODEL_ID,
    system_instruction=EXPANSION_SYSTEM_PROMPT,
)

def check_model_available() -> None:
    try:
        test = gemini_model.generate_content("Hi")
        print(f"✓ Model reachable: {MODEL_ID}")
    except Exception as e:
        raise RuntimeError(
            f"Model '{MODEL_ID}' is not available: {type(e).__name__}: {e}\n"
            "Verify the model ID at aistudio.google.com/models"
        )

check_model_available()

# ── Stage 1: Expansion ───────────────────────────────────────────────────────

def expand_domain_openrouter(domain_text: str, max_retries: int = 3) -> str:
    """Expands the domain using a strict Few-Shot message history and post-generation validation."""
    
    # Trimmed down system prompt (NO EXAMPLES in the text)
    STRICT_SYSTEM_PROMPT = """You are an expert in Sanskrit, Hindu Tantra, Shakta Agamas, and Vajrayana Buddhism. 
        Your ONLY job is to return a comma-separated list of 15-25 Sanskrit nouns and noun phrases that represent the esoteric semantic field of the provided domain.
        STRICT RULES:
        1. Output ONLY nouns and noun phrases.
        2. Include direct Sanskrit equivalents first.
        3. NO formatting, NO bullet points, NO preamble, NO explanation."""

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID, # Use whatever model you are currently testing
                messages=[
                    {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                    
                    # FAKE HISTORY: Example 1
                    {"role": "user", "content": "Fierce Speech Power"},
                    {"role": "assistant", "content": "vak-shakti, vak, speech, divine speech, mantra, mantra-shakti, nada, primordial sound, shabda, shabda-brahman, knowledge transmission, upadesa, sarasvata-vidya, logos"},
                    
                    # FAKE HISTORY: Example 2
                    {"role": "user", "content": "Time and Dissolution"},
                    {"role": "assistant", "content": "kala, mahakala, kalachakra, pralaya, cosmic dissolution, mrityu, death, mahapralaya, tamas, void, shunya, annihilation, cremation ground, shmashanika"},
                    
                    # THE ACTUAL PROMPT
                    {"role": "user", "content": f"Domain to expand: {domain_text}\n\nRETURN ONLY THE COMMA-SEPARATED LIST. NO PREAMBLE."}
                ],
                temperature=0.1, # Extremely low temperature to force rigid formatting
                max_tokens=256
            )
            raw = completion.choices[0].message.content.strip()
            
            # Aggressive cleanup just in case
            raw = raw.replace("\n", " ").strip()
            if raw.lower().startswith("output:"):
                raw = raw[7:].strip()
                
            # ─── POST-GENERATION VERIFICATION (GARBAGE vs FINE) ───────────────
            # 1. Define markers that prove the LLM is hallucinating metadata or formatting
            bad_markers = ["Input:", "Role:", "Domain:", "Output:", "*", "Expert in", "expert in"]
            
            # 2. Check if the response contains any bad markers
            has_bad_formatting = any(marker in raw for marker in bad_markers)
            
            # 3. Check if it actually gave us a comma-separated list (minimum 4 commas for 5 items)
            has_enough_commas = raw.count(",") >= 4
            
            # 4. The Verdict
            if has_bad_formatting or not has_enough_commas:
                print(f"    [Attempt {attempt+1}] ❌ GARBAGE DETECTED. Retrying...")
                print(f"    [Rejected Text]: {raw[:80]}...") # Shows you exactly what failed
                raise ValueError("Validation Failed: Bad LLM Output") 
            else:
                print(f"    [Attempt {attempt+1}] ✅ RESPONSE FINE.")
            # ──────────────────────────────────────────────────────────────────
            
            return raw
            
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error/Retry trigger: {e}")
            import time
            time.sleep(5)
            
    return domain_text


def run_expansion(raw_domains: list, delay: float = 3.0) -> list:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        cached_hits = sum(1 for d in raw_domains if d in cache)
        print(f"Cache loaded  : {len(cache)} total entries")
        print(f"Cache hits    : {cached_hits}/{len(raw_domains)} domains already done")
        print(f"Remaining     : {len(raw_domains) - cached_hits} domains to expand")
    else:
        cache = {}
        print("No cache found — starting fresh.")

    expansions = []
    api_calls  = 0

    for i, domain in enumerate(raw_domains):
        if domain in cache:
            expansions.append(cache[domain])
            continue

        print(f"[{i+1}/{len(raw_domains)}] Expanding: '{domain}'")
        expanded = expand_domain(domain)
        cache[domain] = expanded
        expansions.append(expanded)
        api_calls += 1

        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Done. Waiting {delay}s before next call...")
        time.sleep(delay)

    print(f"\nExpansion complete.")
    print(f"  New API calls this session : {api_calls}")
    print(f"  Total cache entries        : {len(cache)}")
    print(f"  Cache path                 : {CACHE_PATH}")
    return expansions


# ── Stage 2: Phrase-Pooled Embedding ────────────────────────────────────────

def embed_with_phrase_pooling(
    expansions: list,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32,
) -> np.ndarray:
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    all_phrases_per_deity = []
    phrase_to_idx         = {}
    unique_phrases        = []

    for expansion in expansions:
        phrases = [p.strip() for p in expansion.split(",") if p.strip()]
        all_phrases_per_deity.append(phrases)
        for p in phrases:
            if p not in phrase_to_idx:
                phrase_to_idx[p] = len(unique_phrases)
                unique_phrases.append(p)

    print(f"Encoding {len(unique_phrases)} unique phrases across {len(expansions)} deities...")
    phrase_vecs = model.encode(
        unique_phrases,
        show_progress_bar=True,
        normalize_embeddings=False,
        batch_size=batch_size,
    )

    deity_vecs = np.zeros((len(expansions), phrase_vecs.shape[1]))
    for i, phrases in enumerate(all_phrases_per_deity):
        vecs = np.array([phrase_vecs[phrase_to_idx[p]] for p in phrases])
        deity_vecs[i] = vecs.mean(axis=0)

    norms = np.linalg.norm(deity_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    deity_vecs = deity_vecs / norms

    print(f"Embedding matrix shape: {deity_vecs.shape}")
    return deity_vecs


def compute_semantic_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    dist = squareform(pdist(embeddings, metric="cosine"))
    return np.clip(dist, 0.0, 1.0)


# ── Verification ─────────────────────────────────────────────────────────────

def verify_theological_pairs(sem_matrix: np.ndarray, labels: list) -> None:
    pairs = [
        ("Matangi",        "Saraswati",    "CLOSE"),
        ("Nila_Saraswati", "Matangi",      "CLOSE"),
        ("Nila_Saraswati", "Saraswati",    "CLOSE"),
        ("Matangi",        "Kali",         "FAR"),
        ("Saraswati",      "Chhinnamasta", "FAR"),
        ("Nila_Saraswati", "Guhya_Kali",   "FAR"),
        ("Matangi",        "Adi_Shakti",   "FAR"),
    ]

    print("\n── Theological Pair Verification ──────────────────────────────────")
    print(f"  {'Pair':<45} {'Dist':>6}  {'Expected':<8}  Result")
    print(f"  {'─'*45} {'─'*6}  {'─'*8}  {'─'*6}")
    label_list = list(labels)
    for a, b, expectation in pairs:
        if a not in label_list or b not in label_list:
            missing = a if a not in label_list else b
            print(f"  '{missing}' not found in dataset — skipping")
            continue
        i, j = label_list.index(a), label_list.index(b)
        d = sem_matrix[i, j]
        passed = (expectation == "CLOSE" and d < 0.25) or \
                 (expectation == "FAR"   and d > 0.40)
        flag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {a} ↔ {b:<{45 - len(a) - 3}} {d:>6.4f}  {expectation:<8}  {flag}")
    print()


# ── Master function ───────────────────────────────────────────────────────────

def rebuild_semantic_matrix(df: pd.DataFrame, labels: list) -> tuple:
    raw_domains = df["Core_Cosmic_Domain"].fillna("Unspecified").astype(str).tolist()

    print("═" * 58)
    print(f"Step 1: Semantic Field Expansion  ({MODEL_ID})")
    print("═" * 58)
    expansions = run_expansion(raw_domains)

    print("\nSample expansions (first 5):")
    for name, domain, exp in zip(list(labels)[:5], raw_domains[:5], expansions[:5]):
        print(f"\n  {name} | {domain}")
        print(f"  → {exp[:120]}{'...' if len(exp) > 120 else ''}")

    print("\n" + "═" * 58)
    print("Step 2: Phrase-Pooled Embedding  (all-mpnet-base-v2)")
    print("═" * 58)
    embeddings = embed_with_phrase_pooling(expansions)
    sem_matrix = compute_semantic_distance_matrix(embeddings)

    print(f"\nDistance range : [{sem_matrix.min():.4f}, {sem_matrix.max():.4f}]")
    print(f"Mean pairwise  : {sem_matrix[np.triu_indices_from(sem_matrix, k=1)].mean():.4f}")

    verify_theological_pairs(sem_matrix, list(labels))

    df["Domain_Expansion"] = expansions
    return expansions, embeddings, sem_matrix


# ── Execute ───────────────────────────────────────────────────────────────────
print("Starting Google AI Studio expansion pipeline...")
expansions, embeddings, semantic_distance_matrix = rebuild_semantic_matrix(df, labels)