import matplotlib.pyplot as plt
import numpy as np

def plot_comparisons(test_df):
    predicted_negative = {'bert_cnn': [], 'roberta': []}
    predicted_neutral = {'bert_cnn': [], 'roberta': []}
    predicted_positive = {'bert_cnn': [], 'roberta': []}

    false_negative = {'bert_cnn': [], 'roberta': []}
    false_neutral = {'bert_cnn': [], 'roberta': []}
    false_positive = {'bert_cnn': [], 'roberta': []}

   true_negative = {'bert_cnn': [], 'roberta': []}
    true_neutral = {'bert_cnn': [], 'roberta': []}
    true_positive = {'bert_cnn': [], 'roberta': []}

    for idx in range(len(test_df)):
        score = test_df['label'].iloc[idx]
        bert_cnn_result = {
            'bert_cnn_neg': test_df['bert_cnn_predicted'].iloc[idx] == 0,
            'bert_cnn_neu': test_df['bert_cnn_predicted'].iloc[idx] == 1,
            'bert_cnn_pos': test_df['bert_cnn_predicted'].iloc[idx] == 2
        }
        roberta_result = {
            'roberta_neg': test_df['roberta_predicted'].iloc[idx] == 0,
            'roberta_neu': test_df['roberta_predicted'].iloc[idx] == 1,
            'roberta_pos': test_df['roberta_predicted'].iloc[idx] == 2
        }

        if score == 0:
            true_negative['bert_cnn'].append(bert_cnn_result)
            true_negative['roberta'].append(roberta_result)
        elif score == 1:
            true_neutral['bert_cnn'].append(bert_cnn_result)
            true_neutral['roberta'].append(roberta_result)
        elif score == 2:
            true_positive['bert_cnn'].append(bert_cnn_result)
            true_positive['roberta'].append(roberta_result)

        if bert_cnn_result['bert_cnn_neg']:
            predicted_negative['bert_cnn'].append(bert_cnn_result)
            if score > 0:
                false_negative['bert_cnn'].append(bert_cnn_result)
            if roberta_result['roberta_neg']:
                predicted_negative['roberta'].append(roberta_result)
            if score > 0:
                false_negative['roberta'].append(roberta_result)
                

        if bert_cnn_result['bert_cnn_neu']:
            predicted_neutral['bert_cnn'].append(bert_cnn_result)
            if score != 1:
                false_neutral['bert_cnn'].append(bert_cnn_result)
        if roberta_result['roberta_neu']:
            predicted_neutral['roberta'].append(roberta_result)
            if score != 1:
                false_neutral['roberta'].append(roberta_result)
        
        if bert_cnn_result['bert_cnn_pos']:
            predicted_positive['bert_cnn'].append(bert_cnn_result)
            if score < 2:
                false_positive['bert_cnn'].append(bert_cnn_result)
        if roberta_result['roberta_pos']:
            predicted_positive['roberta'].append(roberta_result)
            if score < 2:
                false_positive['roberta'].append(roberta_result)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        data = [
            (true_negative, predicted_negative, false_negative, 'Negatives'),
            (true_positive, predicted_positive, false_positive, 'Positives'),
            (true_neutral, predicted_neutral, false_neutral, 'Neutrals')
        ]

        colors = ['blue', 'green', 'red']

        for ax, (true_data, predicted_data, false_prediction, title) in zip(axes, data):
            counts = {
                'bert_cnn': [len(true_data['bert_cnn']), len(predicted_data['bert_cnn']), len(false_prediction['bert_cnn'])],
                'roberta': [len(true_data['roberta']), len(predicted_data['roberta']), len(false_prediction['roberta'])]
            }
            categories = ['True ' + title, 'Predicted ' + title, 'False ' + title]
            
            x = np.arange(len(categories))
            width = 0.35

            ax.bar(x - width/2, counts['bert_cnn'], width, label='BERT-CNN', color=colors[0])
            ax.bar(x + width/2, counts['roberta'], width, label='RoBERTa', color=colors[1])
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.set_title(f'True vs Predicted vs False {title}')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_df = pd.read_csv('test_predictions.csv')
    plot_comparisons(test_df)