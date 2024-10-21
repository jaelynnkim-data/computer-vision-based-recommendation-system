# [Machine Learning Outfit Styling Recommender System](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/customer-style-recommendation-system-resnet50.ipynb)


### Preface
As sustainable practices in fashion 
emerge as solutions to environmental 
pollution, companies must not only 
develop products using sustainable 
materials, but also appeal to consumers 
trying to utilize existing clothing. 
Consumers often hesitate to buy new 
items if they need to purchase additional 
clothing to make the new item wearable.
Likewise, there is an emerging need for a 
recommendation system that can inform 
an individual on possible ways to style an 
existing item in their closet.


Therefore, the fashion industry must develop strategies that promote the reuse of existing garments while 
designing new products that can easily integrate into consumers' current wardrobes without necessitating the 
purchase of completely new outfits. 
To resolve this need, this project aims to propose a machine learning model that can recognize a user’s preference 
in fashion from their “mood board,” which will be a collection of style inspiration photos that they find 
aesthetically pleasing.


![Deep Learning Model Training Process](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%202.png)


The first step is to train and develop the image-detection model to recognize clothing items from new, unseen 
images. When the mood board is analyzed by this trained model, when the user provides the model with an image 
of an item they are interested in, the model will provide the user with recommendations on ways to style the item



### Exploratory Data Analysis 

#### Identifying Clothing Categories
The initial exploratory data analysis involved identifying all possible unique fashion 
item category names while also making sure to generalize the categories in a fine-tuned manner that would allow for the best image recognition.


This began with examining [the CSV dataset of all product names from the online Zara 
store](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/unique_product_names.csv) to understand the distribution and frequency of different items. This step 
included cleaning and annotating the data to reduce 1972 names to 214 unique 
categories.


![Identifying Clothing Categories](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%203.png)


#### Procuring Real-World Image Data
Using the extracted 214 categories from the Zara product list, 400+ 
images were downloaded directly from Pinterest's search results web page 
using the category names as the search keywords using gallery-dl in 
Python.


![Procuring Real-World Image Data 1](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%204.png)


EDA also involved inspecting the downloaded images for each category to ensure they correctly 
represented the intended clothing items. Image size and quality were assessed to determine 
the need for resizing.
Folders containing images that deviated by a wide degree from one another were deleted from 
the dataset, leaving only 199 categories. This pointed to the fact that the fine-tuned judgement 
on the selection of a category names is critical to retrieve the most accurate results.
Images that clearly did not display the category item were removed by hand to enhance the 
accuracy of the model.


![Procuring Real-World Image Data 2](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%205.png)


### Assumptions About the Data and the Model
#### Assumptions about the Data:
- The 199 clothing categories extracted from the Zara dataset accurately represent a comprehensive range of 
fashion items.
- Images downloaded from Pinterest are accurate representative samples of each clothing category.
- Reducing image sizes to 4KB does not significantly degrade the quality and features necessary for the CNN 
model to perform accurate classification.
- The user-item matrix generated from the image recognition model will effectively capture the presence of 
specific clothing categories in user-uploaded photos.


#### Hypotheses on the Model:
- Given the complexity of the number of categories, the image recognition model will be able to classify clothing 
items into the 199 categories with a relatively low accuracy but will still be able to support the 
recommendation system. 
- The recommendation system will provide relevant clothing combinations based on the user-uploaded images.
- With more fine-tuned cleaning on the raw image data from Pinterest, the image recognition model will be able 
to achieve higher, better accuracy, which will in turn improve the performance of the recommendation system.


### Feature Engineering
The project involved several key feature engineering and 
transformation steps:
- Dataframe Creation: The product names from Zara were converted into a 
dataframe, and unique item names were identified through hand-annotation.
- Image Downloading: Using the 214 clothing categories as keywords, images were 
downloaded from Pinterest into corresponding folders.
- Image Resizing: The downloaded images were resized to 4KB to standardize the 
dataset and reduce computational load.


![Image Resizing](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%206.png)


- Data Normalization and Augmentation: Image data was rescaled and augmented 
with transformations such as shear range, zoom range, and horizontal flips to 
enhance model robustness.
- Test-Train-Validation Data: The images were split to prepare train data, validation 
data and test data for model training. 
- User-Item Matrix: A matrix was created from the CNN model’s classifications of new 
images, representing the presence of each clothing category in the images


<img src="https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%207.png" align="left" height="48" width="48" />

![Validation Data](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%207.png) ![Validation Data](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%208.png)


## Explaining the Approach
### 1. Image Recognition Model
The primary model used was a Convolutional Neural Network (CNN) for image recognition. Within the proposed approach 
of using CNN, three models were checked for overfitting/underfitting as well as test accuracy to determine the best model 
for the proposed solution.


#### [A. The Baseline Model](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/customer-style-recommendation-system-baseline.ipynb)
To prevent overfitting, the baseline model included early stopping based on validation loss, data 
augmentation, and a dropout layer to reduce over-reliance on specific neurons.


![Baseline Model](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%209.png)


##### Interpretation
The large gap between training and validation accuracy, along with training loss decreasing significantly while validation loss does not 
decrease correspondingly, suggests overfitting. The model is learning to perform well on the training data but is not generalizing well to 
unseen data. The relatively flat and low validation accuracy, alongside the minimal decrease or increase in validation loss, indicates that the 
model is not capturing the underlying patterns well enough to perform accurately on new data.


#### [B. The Baseline Model + ResNet50](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/customer-style-recommendation-system-resnet50.ipynb)
Complexity can be increased by adding more layers or using larger pre-trained networks like VGG16, ResNet, 
or MobileNet as feature extractors. ResNet50 was selected and checked for overfitting/underfitting.


![Baseline Model and ResNet50](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2010.png)


##### Interpretation
The model accuracy increases steadily with training, indicating effective learning. However, there's a noticeable gap between training and validation 
accuracy, suggesting some overfitting. The loss decreases for both training and validation, which is good, but the validation loss starts to plateau, 
indicating limitations in learning from the data provided. The baseline model had a higher test accuracy compared to the ResNet50 model. This might 
indicate that the more complex ResNet50 architecture is not able to effectively learn from the dataset or is overfitting to the training data, given that it 
also shows a higher loss.


#### [C. The Baseline Model + L1/L2 Regularization](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/customer-style-recommendation-system_l1l2%20regularization.ipynb)
Regularization techniques can also be used to combat overfitting. L1/L2 regularization was added 
to the baseline model to check for its impact on overfitting/underfitting.


![Baseline Model and L1/L2 Regularization](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2011.png)


##### Interpretation
There is very little change in accuracy, staying extremely low, suggesting that the regularization might be too strong, essentially 
underfitting the model by not allowing it to fit the training data sufficiently. The loss is relatively flat, supporting the notion of 
underfitting. The effects seen from L1/L2 regularization suggest it is too restrictive.


### 2. Recommendation Score Model
Once the CNN model is established, this image recognition model is then used to analyze a set of 100 images from a 
random Pinterest user’s mood board webpage as unseen new data to design a user-item matrix displaying which of 
the categories showed up in each of the 100 images.


![User-Item Matrix](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2012.png)


Cosine similarity and Jaccard similarity were used to examine the most accurate user-item similarity recommendation score, with 
both yielding similar results.


![Cosine Similarity and Jaccard Similarity](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2013.png)


### Model Selection
The selected solution was the baseline CNN model, fine-tuned with early stopping and dropout 
regularization to balance bias and variance. The architecture included multiple convolutional and 
pooling layers to extract features, followed by dense layers for classification. 
L2 regularization was considered but it proved to be too restrictive on training the model with the 
given dataset. 
ResNet50 achieved better shapes on the model accuracy and loss graphs, but the test accuracy being 
significantly lower than the baseline model indicated that the current dataset may not be large and 
complex enough to be able to let the model properly train. 
Therefore, although the selected model was the baseline model with early stopping, the ResNet50 
model was tested alongside the main model to compare results and consider future work. 
Since cosine similarity and Jaccard similarity both yielded similar results, cosine similarity was used to 
perform a user-item approach to build the recommendation score for each fashion item class for each 
image from the user’s mood board.


### Results and Learnings 
The CNN baseline model achieved a low test accuracy of 
0.1462 in classifying clothing items into the 199 categories. 


The recommendation system built on this model suggested 
relevant clothing combinations, but the accuracy was 
compromised as it at times showed a tendency to recognize 
the background color rather than the image’s clothing item. 
As expected, it relied on the input image to have a clean setup 
(clothing item focused stock photo) and high similarity with the 
training data rather than being able to generalize the items in 
the image. 


Likewise, if an image contained a clothing item on a picture of a 
human figure rather than the clothing item alone, the model 
performed poorly to recognize the item. 
Key learnings included the importance of high-quality image 
data, better selection of image data inside categories, the 
effectiveness of data augmentation in improving model 
generalization, and the utility of early stopping to prevent 
overfitting.


![Cosine Similarity and Jaccard Similarity](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2014.png)


##### The case of the light blue satin midi dress
When comparing the recommendation results based on similarity scores from the baseline model side by side with that of the ResNet50 
model, it was noted that the ResNet50 model was recognizing the overall shape and color of the image’s item, while the baseline model was 
able to identify the similar fabric texture and type of item class. 
Despite the fact that the ResNet50 model selected the wrong clothing class, the original baseline model continued to show a lower accuracy 
in being able to correctly identify images from the mood board that contain the exact item displayed in the input photo, while the ResNet50 
model was able to pick out an image from the mood board that matches the given image item almost identically.


![light blue satin midi dress](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2015.png)


The CNN baseline model was also able to suggest other items that are usually seen paired with the 
item in the input image based on the provided mood board using cosine similarity recommendation 
scores.


![Recommendations for Sequined Midi Dress](https://github.com/jaelynnkim-data/customer-style-recommendation-system/blob/main/Recommender%20System%20Image%2016.png)


However, due to the low accuracy of the baseline model’s ability to detect objects in new, unseen images, the clothing 
items in the user’s mood board images were not sorted to reflect the true recommendation for items to go with the 
given input. This limited the effectiveness of the model in practical scenarios where accurate and contextually relevant 
recommendations are crucial. 
By improving the model's ability to understand and classify complex images more effectively, it can be ensured that 
the recommendations are not only more accurate but also tailored to reflect current fashion trends and individual 
preferences, making the system more useful in real-world fashion styling and e-commerce applications.


### Key Takeaways
#### Model Complexity vs. Performance: 
The use of a basic CNN compared to more sophisticated architectures such as ResNet50 highlights the trade-offs between model complexity 
and performance. While simpler models are easier to train and interpret, they may lack the depth required to accurately capture and classify 
more nuanced features in complex image data.


#### Impact of Data Quality and Quantity: 
The project underscores the critical importance of high-quality, diverse training datasets in machine learning. The ability of a model to 
generalize well to new, unseen data is heavily dependent on the variety and representativeness of the training dataset. Enhancements in 
data augmentation and the inclusion of more varied datasets could potentially improve model performance significantly.


#### Evaluation Metrics and Model Tuning: 
The project illustrates the importance of continuous monitoring and evaluation of model performance across various metrics. Understanding 
the behavior of model accuracy and loss during training provides insights into potential issues like overfitting or underfitting, guiding 
necessary adjustments in model architecture.


#### Recommendation System Accuracy: 
The challenges observed with recommendation accuracy highlight the need for models that not only recognize items accurately but also 
understand the context and user preferences. This might involve integrating user profile data, historical interaction data, and even current 
fashion trends into the recommendation system.



### Future Work
- Expanding the dataset to include more images for each 
category while taking time to remove images in folders 
that were downloaded due to misleading search labels 
- Adjusting fashion categories to incorporate any missing 
categories and merge categories that are overlapping 
closely
- Identifying and incorporating image sources that can 
provide more diverse yet accurate image training data
- Finding image sources that can provide more data on 
men’s clothing, given that the primary audience of 
Pinterest consists of a skewed gender identity
- Improving the recommendation algorithm by developing 
a user interface that easily takes input and produces 
output that is immediately recognizable to the eye
- Exploring the fine-tuned versions of the models with 
advanced techniques such as transfer learning with pre-trained models to enhance the object detection model
- Integrating the insight of experts in the fashion or art 
industry for the recommendation system and enhancing 
the model’s training data while taking in the expert input 
- Fine-tuning the ResNet50 model and being able to feed 
the model a larger and more complex dataset to improve 
the test accuracy








