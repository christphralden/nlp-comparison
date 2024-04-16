from transformers import AutoTokenizer as at
from transformers import AutoModelForSequenceClassification as amsc
from scipy.special import softmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm

MODEL_TYPE = f"cardiffnlp/twitter-roberta-base-sentiment"
TOKENIZER = at.from_pretrained(MODEL_TYPE)
MODEL = amsc.from_pretrained(MODEL_TYPE)

#Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text

df = pd.read_csv('./Reviews.csv')
df = df.head(1000)

exmp = df['Text'].values[0]

def polarity_scores_roberta(e):
    encoded_text = TOKENIZER(e, return_tensors='pt')
    output = MODEL(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
        'text':e
    }

    return scores_dict

# EVALUATE
res = {}
df_len = len(df)

for i, row in tqdm(df.iterrows(), total=df_len):
    try:
        text = row['Text']
        myId = row['Id']
        roberta_result = polarity_scores_roberta(text)
        res[myId] = roberta_result
    except RuntimeError:
        print(f'Broke at {myId}')

res_df = pd.DataFrame(res).T
res_df.to_csv('RoBERTa_Result.csv')


# PROCESS
res_df = pd.read_csv('./RoBERTa_Result.csv')
negative_count = 0
false_negative = []

for idx, result in res_df.iterrows():
    if result['roberta_neg'] > result['roberta_pos'] and result['roberta_neg'] > result['roberta_neu']:
        score = df.loc[df['Id'] == idx+1, 'Score'].iloc[0]
        # print(f"Id: {idx} - Negative Sentiment: {result['roberta_neg']} - Score: {score}")
        # print(f"- Text: {result['text']}")
        negative_count += 1
        if score > 3:
            print(f"Negative Sentiment: {result['roberta_neg']}\nActual Score: {score}\nText:{result['text']}\n\n")
            false_negative.append(result)

print(f'Negative Count:{negative_count}, False Negative (>3):{len(false_negative)}, Percentage: {(len(false_negative)/negative_count)*100}')


# PLOT
def plot():
    evaluate_data = [len(res_df), negative_count, len(false_negative)]
    labels = ['Total Data', 'Negative Sentiment', 'False Negatives (>3)']

    plt.figure(figsize=(8, 4))
    plt.bar(labels, evaluate_data, color=['green', 'blue', 'red'])

    plt.title('roBERTa')
    plt.xlabel('Categories')
    plt.ylabel('Count')

    plt.show()

plot()