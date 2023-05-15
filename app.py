import io
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import base64

# Initialize the Dash app
app = Dash(__name__)

# Layout of the Dash app
app.layout = html.Div(children=[
    html.H1(children='Sentiment Word Clouds', style={'textAlign': 'center'}),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '30%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0 auto'
        }
    ),
    
    dcc.Loading(
        id='loading',
        type='circle',
        children=[
            html.Div(id='output-data-upload')
        ],
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100%'}
    )
])

# Callback function to process the uploaded file and perform sentiment analysis
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        # Read the CSV file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return html.Div([
                'There was an error processing this file.'
            ])
        
        # Perform sentiment analysis
        sentiments = []
        for title in df['title']:
            blob = TextBlob(title)
            sentiment = blob.sentiment.polarity
            if sentiment < -0.6:
                sentiment_label = 'Very Negative'
            elif sentiment < -0.2:
                sentiment_label = 'Negative'
            elif sentiment <= 0.2:
                sentiment_label = 'Neutral'
            elif sentiment <= 0.6:
                sentiment_label = 'Positive'
            else:
                sentiment_label = 'Very Positive'
            sentiments.append({'title': title, 'sentiment': sentiment, 'sentiment_label': sentiment_label})
        
        # Extract positive and negative titles
        positive_titles = [sentiment['title'] for sentiment in sentiments if sentiment['sentiment'] > 0 and sentiment['title'] is not None]
        negative_titles = [sentiment['title'] for sentiment in sentiments if sentiment['sentiment'] < 0 and sentiment['title'] is not None]
        
        # Generate word clouds for positive and negative titles
        if positive_titles:
            positive_text = ' '.join(positive_titles)
            positive_wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(positive_text)
            fig_positive = px.imshow(positive_wordcloud.to_array())
            fig_positive.update_layout(title_text='Positive Sentiment Word Cloud')
            
        if negative_titles:
            negative_text = ' '.join(negative_titles)
            negative_wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(negative_text)
            fig_negative = px.imshow(negative_wordcloud.to_array())
            fig_negative.update_layout(title_text='Negative Sentiment Word Cloud')
        
        # Read the CSV file
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return html.Div([
                'There was an error processing this file.'
            ]), {'display': 'none'}
        
        # Perform sentiment analysis
        sentiments = []
        for title in df['title']:
            blob = TextBlob(title)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        
        # Sentiment distribution
        sentiment_distribution = {
            'Very Negative': len([s for s in sentiments if s < -0.6]),
            'Negative': len([s for s in sentiments if -0.6 <= s < -0.2]),
            'Neutral': len([s for s in sentiments if -0.2 <= s <= 0.2]),
            'Positive': len([s for s in sentiments if 0.2 < s <= 0.6]),
            'Very Positive': len([s for s in sentiments if s > 0.6])
        }
        
        # Generate sentiment distribution chart
        fig_sentiment = px.bar(
            x=list(sentiment_distribution.keys()),
            y=list(sentiment_distribution.values()),
            labels={'x': 'Sentiment', 'y': 'Count'},
            title='Sentiment Distribution'
        )
        
        # Return the word cloud figures and the sentiment analysis pie chart
        return [
          html.Div(
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'minWidth': '0px',
                        'maxWidth': '180px',
                        'whiteSpace': 'normal'
                    }
                ),
                style={'margin': '20px', 'overflowX': 'auto'}
            ),
            html.H2(children='Positive Sentiment'),
            dcc.Graph(figure=fig_positive),
            html.H2(children='Negative Sentiment'),
            dcc.Graph(figure=fig_negative),
            html.H2(children='Sentiment Analysis'),
            dcc.Graph(figure=fig_sentiment)
        ]

if __name__ == '__main__':
    app.run_server(debug=True)
