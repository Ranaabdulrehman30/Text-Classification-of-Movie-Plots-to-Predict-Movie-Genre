Business Understanding
Movies are a popular means of entertainment all around the world. With a lot of streaming services and websites available and with more on the horizon, building a predictive model that automatically generates the genre of a movie based on the its plot summary is definitely beneficial as it may reduce the expense of manually tagging the movies.

File 1: DNNMODEL_using_plot.ipynb
This initial model was built to check the basic performance of a DNN model with very less preprocessing of the data. Since we are working with textual data, understanding the representation of data is generally a hard task at hand. Once we encoded the Classes a sequential model was built with multiple dense layers and the number of neurons vary. Once you train the model on (train_data) and test it on (test_data). The accuracy for training turns out to be close to 98% but the test accuracy turns out to be accord 28%. The model is clearly Overfitting. 
This is because very little preprocessing of data was done i.e the plot of the movies contains a lot of stopwords and repeated words.

File 2: predict_movie_genre_from_plot.ipynb
The dataset used in this notebook was obtained from AWS s3 bucket and it contains movie plot summaries scraped from IMDB. The dataset has 4610 entries. Information found in the dataset  is as follows:
Release Year - year of release
Title - title of the movies
Director - director names associated with the movies
Cast - cast name associated with the movies
Movie poster
Plot - plot summary of the movies etc

For these models we started off with the exploration of data which includes descriptive statistics and the snippets of the results are in the code file. Furthermore data was cleaned and unnecessary columns were dropped such as movie-rating, ranking, director_info etc.

This model was built with a different approach rather than predicting all the genres, we have selected top 6 genres and trained the model to predict these them. We then tried to clean the text of the plot.
Creating our stopwords list and adding some more words that are very common in the summaries. (I ran the freq_list after this and noticed that some of these words were very common but didn't lend us too much meaning so I came back here to add them to the stopwords list). Changing the text to lower case, stopwords removal, lemmatizing. Visualization of the comparison of total and unique number of words before and after processing is in the code.
Further description of the applied model and their results is in the code.

Comparison of the Different Models and their Scores

The best model is the one that gave us the best score above which is Grid Search with Multinomial Naive Bayes on the tfidf-transformed X variable which gave us an accuracy score of approximately 61%. Using the '.best_params_' attribute of GridSearchCV, I will obtain the optimal hyparameter values and use it in evaluating the test set.

Conclusion

Results

Baseline accuracy was 41%
The tfidf transformed plots performed better during modeling.
GridSearchCV helped in narrowing down the best model hyperparameter values.
The model with the best performance was the Grid Search with Multinomial Naive Bayes on tfidf transformed plots with the following parameters (although SGD ranked pretty highly as well):
'alpha': 0.001, 'class_weight': None, 'loss': 'log', 'max_iter': 10, 'penalty': 'l2'}
When the above model was used on the test set, it produced an accuracy score of ~61% which a significant increase from the baseline
The confusion matrix confirms what I suspected when looking at the most common words in the genres.
Due to the fact that some of the genres had words in common with another genre, false predictions of those genres were mostly as the genres they had common words with.
The decision tree models performed the least favorably.


These are not the final models. Since we have made progress in making the initial Model, more work will be done to make a high performing model.
