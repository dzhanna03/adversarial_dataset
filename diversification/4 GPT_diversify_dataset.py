from openai import OpenAI
import pandas as pd

client = OpenAI(api_key='')

def paraphrase_sentences(df):
    paraphrased_sentences = []
    skipped_indices = []

    for i, row in df.iterrows():
        if (i + 1) % 3 == 0:
            paraphrased_sentences.append(row['premise'])
            skipped_indices.append(i)
            continue 

        messages = [
            {"role": "system", "content": "You are a skilled linguist who paraphrases sentences. Your goal is to creatively use synonyms or hyponyms from the same hypernym category, add appropriate adjectives or adverbs to enrich the sentences. You must very strictly preserve the original verbs and their forms. Use frequently used words in everyday English. The paraphrased sentences must remain grammatically correct, plausible, logical."},
            {"role": "user", "content": f"Paraphrase the sentence without changing the verbs: {row['premise']}"}
        ]
        
        completion = client.chat.completions.create(
          model="gpt-4-0125-preview",
          messages=messages
        )

        paraphrased_sentence = completion.choices[0].message.content
        paraphrased_sentences.append(paraphrased_sentence)

    return pd.DataFrame({
        'premise': df['premise'],
        'changed_premise': paraphrased_sentences
    }), skipped_indices

df = pd.read_csv('C:/Users/adzhu/thesis_prep/3 premises_tenses.csv')
new_df, skipped = paraphrase_sentences(df)
new_df.to_csv('C:/Users/adzhu/thesis_prep/4 premises_paraphrased.csv', index=False)

