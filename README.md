Kelly Liu, Shan Ming Gao, CP Chan, Melody Chang

#  Motivation and Problem Definition
Cyberbullying as defined by the American Psychological Association is the “willful and repeated harm inflicted through the use of computers, cell phones, and other electronic devices.” (Abramson, 2022). Being a victim of cyberbullying is associated with depression, anxiety, suicidal thoughts and attempts, and more. In 2023, a study of 26.5% of 5,000 nationally representative, middle and high schoolers reported being bullied in the last 30 days; this statistic has also risen, from 23.2% in 2021 and 17% in 2019 (Hinduja and Patchin, 2024). 

Thus, cyberbullying detection, understanding, protection, and prevention at scale are highly valuable for public mental health. Our data science project contributes to the foundational detection and understanding components.

Problem Statement: How may we create an unimodal cyberbullying detection model that incorporates text analysis in predicting the presence of cyberbullying and identifying the social context associations of the messages?

## Project Questions
1. Understand cyber-bullying dataset context by using natural language processing techniques to do text preprocessing and sentiment analysis
2. Cyberbullying detection model (unimodal) to the prediction of if the context is bully or not, and which category the content falls into
3. Test the prediction model's accuracy, precision, recall, and F1 score using the confusion matrix

# Related work (Directly related)
Cyberbullying detection with machine learning has been practiced before, and deep neural networks in particular are more effective than conventional techniques (Raj et al., 2022). Previous work on cyberbullying detection has also considered other factors, such as cyberbullying role detection (ie. defenders, bystanders, and instigators), non-textual cyberbullying (eg. with OCR classification) (Logasree & Harshini, 2023), and personality associations (Balakrishnan et al, 2020).

# Methodology (Data, Transformation, and Evaluation) & Results
Our work is made possible by the efforts of Ahmadinejad et al. (2023) who generated a dataset of 99,991 Tweets (Twitter/X), with one label for non-cyberbullying and three for cyberbullying of racial, religious, or gender-and-sexuality nature. Significantly, their labels were verified by randomly sampling 1000 tweets and giving them to three “social media specialists with experience in detecting cyberbullying” - ie. domain experts - who affirmed over 90% classification accuracy.

Among all metrics, Recall will be the most important metric for us, the less false negative the better. We do not want our model to see a tweet that is supposed to be cyberbullying content and think that it is not. The feature input will be the word embedding which is converted from the raw tweet.

For the text preprocessing, we use Natural Language Toolkit NLTK -Tokenization and Stopwords to remove common words lacking significant meaning. Then we load the pre-trained word2vec model from Google News with gensim. downloader. After having the pre-trained word2vec model, we convert the raw tweet to word embeddings, which is the numeric vector input. The distance and direction between vectors indicate the similarity and relationships between words. This will be the feature we used for all the detection models. 

## A. Text and Sentiment Analysis
In the text-cleaning phase, we will create four lists based on the categories on the label. Following categorization, we will conduct frequency analysis to extract insights from the most commonly occurring words and contexts within each category. Additionally, we will use word clouds for visual representation to clearly show the “keywords” in each category.

Figure 1. Word Clouds by Category
Note: Due to the nature of our topic, many of the words below are highly offensive.
![Screenshot 2024-07-01 at 6 16 06 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/51a9a72e-b792-46dc-a55b-86cf7d1a9fac)

Table 1. Sentiment Analysis Results
![Screenshot 2024-07-01 at 6 16 54 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/0876e223-042f-4042-bad5-533278dbccfe)

## B. Cyberbullying Detection Model

### 1. MLPClassifier
We chose Multi-Layer Perceptrons (MLP) as one of the models for cyberbullying detection. For the hidden layers, we chose 4 hidden layers {200,100,50,25}, with the layers from bigger to smaller size. The reason that we set the hidden layers this way, is that with a bigger size on the first layer, and a gradually smaller size, it can help the network learn a hierarchy of features, where more features are captured in the wider layers and more detailed features in the narrower layers.

From the results, it has a high Recall value, indicating that MLPClassifier is doing a good job of cyberbullying detection. Based on the Recall, this MLPClassifier model performed well on “Not cyberbullying” and “ethnicity/race” content. 0.98 for “Not cyberbullying” in Recall, and 0.98 for “ethnicity race”. 
![Screenshot 2024-07-01 at 6 21 43 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/e1aceeb2-0276-416d-9db9-7a168cef5f1b)

### 2. TensorFlow
For the TensorFlow, we start with the label encoder which turns categorical labels into numerical labels. After splitting training and testing sets, we can build the sequential model by adding dense and dropout layers to prevent overfitting. We can use test loss to see how well a machine learning model performs on data that it has not seen before, and identify whether it is overfitting or underfitting. 

The model has a low test loss, indicating that it is not overfitting or underfitting. It also has a high Recall value just like the MLPClassifier, indicating that it is doing a good job on cyberbullying detection. We then can look into the performance of each cyberbullying and non-cyberbullying category. Based on the Recall, this Tensorflow model performed well on “Not cyberbullying” and “religion” content. 0.99 for “Not cyberbullying” in Recall, and 0.97 for “religion”.
![Screenshot 2024-07-01 at 6 22 41 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/36ff61ef-f241-4866-aa6f-7d853b02247a)

### 3. Logistic Regression
The model is set with the solver as 'Liblinear' to handle highly sparse data and includes regularization to prevent overfitting on the training data. It shows better precision overall as well as classifying and identifying 'Non-cyberbullying' contents. However, it struggles the most when it comes to classifying positive instances within the 'ethnicity/race' category.
![Screenshot 2024-07-01 at 6 23 16 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/06cfbe91-2f69-47ea-b749-8b93ff9404d4)

### 4. SVM
SVM operates effectively in N-dimensional space and can handle non-linear relationships. The kernel setting for the SVM classification model is RBF, since its superior performance in text or image categorization tasks. Results indicate the model can almost perfectly categorize religion-related bullying content and reliably detect non-cyberbullying content. However, instances of bullying based on ethnicity/race are more likely to be overlooked by the model.
![Screenshot 2024-07-01 at 6 24 20 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/603b0ab8-55e8-45d6-8cbf-889c46952117)

# Key Findings and Conclusion
Sentimental Analysis: Based on polarity scores, all cyberbullying content exhibits a negative sentiment, with the ethnicity/race group showing the most negativity, while non-cyberbullying content receives a positive sentiment.

Text Analysis: Black people and women are major targets of cyberbullying. Terms like the "n-word" and the "b-word" frequently surface in the ethnicity/race category. These two groups have been historically marginalized, and unfortunately, this reality persists today. Insults aimed at women appear in the gender/sexual category as well. We see the "b-word" once again, alongside "women," "rape," and the "f-word", highlighting women continue to be viewed as vulnerable targets of disrespect and crime. In the religion category, 'Muslim' is at the top of the cyberbullying list, illustrating the lasting impact of 911. In non-cyberbullying content, the most frequent words are 'love' and 'time', with many of the words reflecting neutral or positive sentiments.

Prediction Models: MLP Classifier, Tensorflow, and SVM all achieved high scores of 0.98 across accuracy, precision, recall, and F1 Score. On the other hand, Logistic Regression achieved slightly lower scores of 0.94 across all metrics, this could be due to it works on linear relationships only. 

Two other key findings observed, first, all the models have stronger ability to capture non-cyberbullying content as indicated by higher recall values (Table 2). This could be because of imbalance dataset, there are much more data on non-cyberbullying content than on other types. Secondly, most models struggle to identify all instances of bullying content related to ethnicity or race, as it has the lowest recall values in general. This could be because many keywords overlap with gender/sexual, making it harder for models to detect. 
![Screenshot 2024-07-01 at 6 25 23 PM](https://github.com/kellyhuanyu/Unimodal-Cyberbullying-Detection-and-NLP-Analysis/assets/105426157/ec4f7b1c-df76-46f4-8fd1-7054dcc2a073)

# Ethical Limitations and Future Work 
There are several limitations to the model and analysis of our work, which also suggest potential future directions. One major limitation is unimodality; social media content often extends beyond text, encompassing images, videos, and interactions that are not captured in isolated tweets or words. Additionally, the dataset's nearly 100k tweets are better balanced as two classes (cyberbullying or not;  vs 9916 vs 10082 observations, respectively) than when analyzed as four classes due to the dataset’s structure. However, data imbalances are also reflected in real-world scenarios, where only about 10% of tweets encountered by Ahmadinejad et al. (2023) were classified as cyberbullying. Moreover, the dataset's limitation to four classes oversimplifies the complexity of real-world behavior where topics are not mutually exclusive and often overlap. This especially implicates the concept of intersectionality in multi-topic cyberbullying: for instance, a Tweet targeting Black women would need to be classed race or gender (not both), reducing the accurate representation of misogynoir in the real world. Furthermore, the three cyberbullying classes do not cover all legally protected class categories, such as disability, age, and political orientation, with user age being particularly crucial in the context of cyberbullying. Future work could expand on existing research by incorporating classifications based on cyberbullying roles, psychometric and personality data, and user behavior to create a more comprehensive and nuanced understanding of cyberbullying on social media.








