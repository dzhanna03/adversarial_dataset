from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=' ')

def paraphrase_sentences(df):
    paraphrased_sentences = []
    skipped_indices = []

    for i, row in df.iterrows():
        if (i + 1) % 3 == 0:
            paraphrased_sentences.append(row['2changed_premise'])
            skipped_indices.append(i)
            continue  

        messages = [
            {"role": "system", "content": "You are a skilled linguist, and your goal is to creatively add attributes and adverbial modifiers to the sentences. Use frequently used words in everyday English. The edited sentences must remain grammatically correct, plausible, logical."},
            {"role": "user", "content": f"Creatively add attributes and adverbial modifiers to the sentences: {row['2changed_premise']}"}
        ]
        
        completion = client.chat.completions.create(
          model="gpt-4-0125-preview",
          messages=messages
        )

        paraphrased_sentence = completion.choices[0].message.content
        paraphrased_sentences.append(paraphrased_sentence)

    return pd.DataFrame({
        '2changed_premise': df['2changed_premise'],
        '3changed_premise': paraphrased_sentences
    }), skipped_indices

df = pd.read_csv('C:/Users/adzhu/thesis_prep/6 premises_substituted_values.csv')
new_df, skipped = paraphrase_sentences(df)
new_df.to_csv('C:/Users/adzhu/thesis_prep/7 premises_paraphrased.csv', index=False)

