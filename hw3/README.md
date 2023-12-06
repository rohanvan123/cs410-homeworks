# Homework Solutions

### Question 1

Here are the results for question 1:

**Including stop words**:

best activation function is relu

<img src="./images/output_plot_sw.png"/>

longest doc: ['possible', 'playoff', 'preview', 'a', 's', 'take', 'a', 'hit', 'in', 'toronto', 'but', 'come', 'home', 'lt', 'b', 'gt', 'lt', 'gt']

Two predicted predicted_words from longest doc are ['lt', 'lt']

Actual: ['manchester', 'hotspur', 'all', 'us', 'cereals', 'to', 'whole', 'grain', 'iraq', 'detainees', 'to', 'play', 'at', 'wrigley', 'field', 'ap', 'hmmm', 'hmmm', 'good', 'lip']

Predicted: ['to', 'with', 'to', 'with', 'with', 'with', 'with', 'with', 'with', 'and', 'with', 'with', 'profit', 'from', 'with', 'with', 'with', 'and', 'with', 'with']

Our ouput has an f1 score of 0, with precision of 0.0 and recall of 0.0

**Not including stop words**:

best activation function is relu

<img src="./images/output_plot_nsw.png"/>

longest doc: ['afp', 'interview', 'un', 'refugee', 'chief', 'says', 'sudan', 'likely', 'grant', 'darfur', 'lt', 'b', 'gt', 'lt', 'gt']

Two predicted predicted_words from longest doc are ['lt', 'lt']

Actual: ['manchester', 'hotspur', 'us', 'cereals', 'whole', 'grain', 'detainees', 'play', 'wrigley', 'field', 'ap', 'hmmm', 'hmmm', 'good', 'lip', 'charles', 'jousts', 'critics', 'game', 'tech']

Predicted: ['us', 'us', 'us', 'us', 'killed', 'killed', 'us', 'us', 'us', 'us', 'us', 'us', 'us', 'us', 'killed', 'killed', 'us', 'us', 'us', 'us']

Our ouput has an f1 score of 0.00019654088050314464, with precision of 0.
00029481132075471697 and recall of 0.00014740566037735848

**Conclusion**:

Because the model trained without stopwords has a slightly better f1_score (albiet still very small), without stopwards appears to be the more accurate form.

### Question 2
