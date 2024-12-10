# SentimentAnalysis-Flipcart-Reviews
This project performs sentiment analysis on Flipkart product reviews to classify customer sentiments into categories such as Positive, Negative, or Neutral. The analysis is based on customer feedback data.

# Project Workflow
1. **Data Preprocessing**:
    - Removal of missing values.
    - Splitting data into training and testing sets.
    - Text vectorization using `CountVectorizer`.
    
2. **Model Training**:
    - A Multinomial Naive Bayes classifier is trained on the vectorized review summaries.

3. **Evaluation**:
    - Performance metrics such as accuracy and a classification report are generated.

4. **Visualization**:
    - Sentiment distribution bar chart.
    - Word cloud of frequently used words in reviews.

