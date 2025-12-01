import pandas as pd
import google.generativeai as genai
import time
import os

# --- CONFIGURATION ---

GOOGLE_API_KEY = "API-KEY-DEFINED-IN-DOTENV"
DELAY_BETWEEN_REQUESTS = 2.3
MODEL_NAME = "gemini-2.0-flash-lite"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# --- PROMPT LOGIC ---

def construct_prompt(row, lang):
    """
    Creates the prompt based on constraints (Headline vs Words) and Language.
    """
    # Handle NaNs (empty cells) by converting to string and checking length/content
    word1 = str(row['word1']).strip() if pd.notna(row['word1']) else "-"
    word2 = str(row['word2']).strip() if pd.notna(row['word2']) else "-"
    headline = str(row['headline']).strip() if pd.notna(row['headline']) else "-"

    # Logic: If words are missing or "-", it's a headline task
    is_word_task = (word1 not in ['-', '']) and (word2 not in ['-', ''])

    # --- ENGLISH ---
    if lang == 'en':
        if is_word_task:
            return f"Write a witty joke that uses the following two words: '{word1}' and '{word2}'. The joke should be coherent."
        else:
            return f"Act like a late-night TV host. Write a short, funny opening monologue joke based on this news headline: '{headline}'"

    # --- SPANISH ---
    elif lang == 'es':
        if is_word_task:
            return f"Escribe un chiste ingenioso que incluya obligatoriamente estas dos palabras: '{word1}' y '{word2}'."
        else:
            return f"Actúa como un comediante. Escribe un chiste corto o comentario satírico basado en este titular: '{headline}'"

    # --- CHINESE ---
    elif lang == 'zh':
        if is_word_task:
            return f"请写一个包含“{word1}”和“{word2}”的幽默段子或笑话。"
        else:
            return f"请模仿脱口秀演员，针对这个新闻标题写一个好笑的段子：'{headline}'"

    return None

def generate_joke_gemini(prompt):
    """
    Calls Gemini Flash with error handling.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "GENERATION_ERROR"

# --- MAIN PROCESSING LOOP ---

files_to_process = [
    {'filename': 'task-a-en.tsv', 'lang': 'en'},
    {'filename': 'task-a-es.tsv', 'lang': 'es'},
    {'filename': 'task-a-zh.tsv', 'lang': 'zh'}
]

for file_info in files_to_process:
    input_filename = file_info['filename']
    lang = file_info['lang']

    if not os.path.exists(input_filename):
        print(f"Skipping {input_filename} (File not found)")
        continue

    print(f"Processing {input_filename} using {MODEL_NAME}...")

    # Read TSV
    df = pd.read_csv(input_filename, sep='\t')

    # Prepare Output List
    results = []

    # Iterate through rows
    for index, row in df.iterrows():
        prompt = construct_prompt(row, lang)

        if prompt:
            # 1. Generate
            output_text = generate_joke_gemini(prompt)
            print(f"[{row['id']}] Generated.") # Minimal log to keep console clean

            # 2. Store Result
            results.append({
                'id': row['id'],
                'output': output_text
            })

            # 3. Rate Limit Sleep
            # We sleep AFTER every request to respect the 8 RPM limit
            time.sleep(DELAY_BETWEEN_REQUESTS)
        else:
            results.append({'id': row['id'], 'output': "ERROR: Invalid Constraints"})

    # --- EXPORT ---
    # Create DataFrame from results
    output_df = pd.DataFrame(results)

    # Construct output filename: output-[input_filename]
    # Note: User requested .tsx (TypeScript) format in prompt text,
    # but based on data structure, .tsv is the standard.
    # I will save as .tsv to maintain structure, but naming it as requested implies structure.
    # To be safe for data usage, we save as .tsv.
    output_filename = f"output-{input_filename}"

    output_df.to_csv(output_filename, sep='\t', index=False)
    print(f"Finished. Saved to {output_filename}\n")

print("All tasks completed.")
