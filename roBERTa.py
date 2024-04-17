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
df = df.head(10000)
# df.set_index('Id', inplace=True)

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
# res = {}
# df_len = len(df)

# for i, row in tqdm(df.iterrows(), total=df_len):
#     try:
#         text = row['Text']
#         myId = row['Id']
#         roberta_result = polarity_scores_roberta(text)
#         res[myId] = roberta_result
#     except RuntimeError:
#         print(f'Broke at {myId}')

# res_df = pd.DataFrame(res).T
# res_df.to_csv('RoBERTa_Result.csv')


# PROCESS
res_df = pd.read_csv('./RoBERTa_Result.csv') # Id nya harus tambahin manual gw males

predicted_negative = []
predicted_neutral = []
predicted_postivie = []

false_negative = []
false_neutral = []
false_positive = []

true_negative = []
true_neutral = []
true_positive = []

for idx, result in res_df.iterrows():
    score = df.loc[df['Id'] == idx+1, 'Score'].iloc[0]
    #we will assume that 3 is neutral
    

    if score < 3: # case actually negative
        true_negative.append(result)
    elif score == 3: # case actually neutral
        true_neutral.append(result)
    elif score > 3 :
        true_positive.append(result)


    if result['roberta_neg'] > result['roberta_pos'] and result['roberta_neg'] > result['roberta_neu']: # case where negative is dominant
        predicted_negative.append(result)
        if score > 3: # false negatives
            false_negative.append(result)
    elif result['roberta_neu'] > result['roberta_neg'] and result['roberta_neu'] > result['roberta_pos']:
        predicted_neutral.append(result)
        if score != 3:
            false_neutral.append(result)
    elif result['roberta_pos'] > result['roberta_neg'] and result['roberta_pos'] > result['roberta_neu']:
        predicted_postivie.append(result)
        if score < 3:
            false_positive.append(result)
        

def delta(true_values, pred_values):
    true_set = set(true_values)
    fake_predictions = []
    for predicted in pred_values:
        if predicted not in true_set:
            fake_predictions.append(predicted)
    return fake_predictions

def plot_all_comparisons(true_negatives, predicted_negatives, false_negatives, true_positives, predicted_positives, false_positives, true_neutrals, predicted_neutrals, false_neutrals):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of subplots

    data = [
        (true_negatives, predicted_negatives, false_negatives, 'Negatives'),
        (true_positives, predicted_positives, false_positives, 'Positives'),
        (true_neutrals, predicted_neutrals, false_neutrals,'Neutrals')
    ]
    
    colors = ['blue', 'green', 'red']

    for ax, (true_data, predicted_data, false_prediction, title) in zip(axes, data):
        categories = ['True ' + title, 'Predicted ' + title, 'False ' + title]
        counts = [len(true_data), len(predicted_data), len(false_prediction)]
        ax.bar(categories, counts, color=colors)
        ax.set_title('True vs Predicted vs False ' + title)
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(counts) + 10)

    plt.tight_layout()
    plt.show()


plot_all_comparisons(true_negative, predicted_negative, false_negative, true_positive, predicted_postivie, false_positive, true_neutral, predicted_neutral, false_neutral)