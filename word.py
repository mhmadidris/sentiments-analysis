import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template
from textblob import TextBlob

app = Flask(__name__)

df = pd.read_csv('Indonesia-Berita.csv')

df.head()

def get_sentiment_analysis_from_csv():
    sentiments = []
    for index, row in df.iterrows():
        title = str(row['title'])
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity
        sentiments.append({'title': title, 'sentiment': sentiment})
    return sentiments

@app.route('/')
def index():
    # Perform sentiment analysis on the news titles
    sentiments = get_sentiment_analysis_from_csv()

    # Convert sentiments to a pandas DataFrame
    df2 = pd.DataFrame(sentiments)

    # Group sentiments based on the range of sentiment values (e.g., 'sentiment_group')
    bins = [-1, -0.5, 0, 0.5, 1]
    labels = ['Very Negative', 'Negative', 'Positive', 'Very Positive']
    df2['sentiment_group'] = pd.cut(df2['sentiment'], bins=bins, labels=labels)

    # Count the number of news articles in each sentiment group
    sentiment_counts = df2['sentiment_group'].value_counts()

    # Create the bar chart using Plotly
    fig = go.Figure(data=go.Bar(x=sentiment_counts.index, y=sentiment_counts.values))

    fig.update_layout(
        title='Sentiment Analysis News',
        xaxis_title='Sentiment Group',
        yaxis_title='Number of News',
        template='plotly_white'
    )

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    # Render the template and pass the plot HTML
    return render_template('index.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
