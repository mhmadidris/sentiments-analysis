import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request
from textblob import TextBlob
from tqdm import tqdm
from wordcloud import WordCloud
from io import BytesIO
import base64

app = Flask(__name__)

def get_sentiment_analysis_from_csv(file):
    df = pd.read_csv(file)
    sentiments = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Performing Sentiment Analysis'):
        title = str(row['title'])
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity
        sentiments.append({'title': title, 'sentiment': sentiment})
    return sentiments

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            sentiments = get_sentiment_analysis_from_csv(file)
            df2 = pd.DataFrame(sentiments)
            bins = [-1, -0.5, 0, 0.5, 1]
            labels = ['Very Negative', 'Negative', 'Positive', 'Very Positive']
            df2['sentiment_group'] = pd.cut(df2['sentiment'], bins=bins, labels=labels)
            sentiment_counts = df2['sentiment_group'].value_counts()
            fig = go.Figure(data=go.Bar(x=sentiment_counts.index, y=sentiment_counts.values))
            fig.update_layout(
                title='Sentiment Analysis News',
                xaxis_title='Sentiment Group',
                yaxis_title='Number of News',
                template='plotly_white'
            )
            plot_html = fig.to_html(full_html=False)

            # Perform word cloud visualization for positive and negative sentiment words
            positive_titles = [sentiment['title'] for sentiment in sentiments if sentiment['sentiment'] > 0 and sentiment['title'] is not None]
            negative_titles = [sentiment['title'] for sentiment in sentiments if sentiment['sentiment'] < 0 and sentiment['title'] is not None]

            # Generate word cloud for positive sentiment words
            if positive_titles:
                positive_text = ' '.join(positive_titles)
                positive_wordcloud = WordCloud().generate(positive_text)
                buffer = BytesIO()
                positive_wordcloud.to_image().save(buffer, format='PNG')
                positive_wordcloud_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Generate word cloud for negative sentiment words
            if negative_titles:
                negative_text = ' '.join(negative_titles)
                negative_wordcloud = WordCloud().generate(negative_text)
                buffer = BytesIO()
                negative_wordcloud.to_image().save(buffer, format='PNG')
                negative_wordcloud_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Generate heatmap
            heatmap_data = df2.pivot_table(index='sentiment_group', columns='sentiment_group', values='sentiment', aggfunc='count', fill_value=0)
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=labels,
                y=labels,
                colorscale='RdYlGn',
                zmin=0,
                zmax=heatmap_data.values.max(),
                colorbar=dict(title='Number of News')
            ))
            fig_heatmap.update_layout(
                title='Sentiment Analysis Heatmap',
                xaxis_title='Sentiment Group',
                yaxis_title='Sentiment Group',
                template='plotly_white'
            )
            heatmap_html = fig_heatmap.to_html(full_html=False)

            return render_template('index.html', plot_html=plot_html, positive_wordcloud_encoded=positive_wordcloud_encoded, negative_wordcloud_encoded=negative_wordcloud_encoded, heatmap_html=heatmap_html)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
