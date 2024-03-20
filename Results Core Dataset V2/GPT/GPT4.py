from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=' ')

def entailment(premise, hypothesis):
    prompt = (f"Given the premise: \"{premise}\" "
              f"and the hypothesis: \"{hypothesis}\", "
              "It is a Natural Language Inference task which determines if one sentence entails another. You have to decide whether a hypothesis can be inferred from a given premise. Provide 'entailment' (when the hypothesis can be inferred from the given premise) or 'non-entailment' (when the hypothesis cannot be inferred from the given premise) as your answer. You can use only two words 'entailment' and 'non-entailment' without any additional text/explanations, without capital letters and punctuation marks.")

    completion = client.chat.completions.create(
      model="gpt-4-0125-preview",
      messages=[
        {"role": "system", "content": "You are a helpful assistant that knows everything about Natural Language Inference and determines if the hypothesis is an entailment or non-entailment of the premise. You use only two words 'entailment' and 'non-entailment'."},
        {"role": "user", "content": prompt}
      ]
    )

    return completion.choices[0].message.content

def process_dataset(input_file, output_file):
    df = pd.read_csv(input_file)

    predictions = [entailment(row['premise'], row['hypothesis']) for _, row in df.iterrows()]

    df['prediction'] = predictions

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    input_file = 'C:/Users/adzhu/thesis_prep/dataset_core_v2.csv'
    output_file = 'C:/Users/adzhu/thesis_prep/GPT4_pred_v2_3.csv'
    process_dataset(input_file, output_file)
