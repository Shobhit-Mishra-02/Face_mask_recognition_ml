# Face_mask_recognition_ml
Here I have developed a ml model which will allow the user to detect a face and also it will tell you that whether that detected face has got a face mask or no.
<h1>Project for the face mask indentification with ml.</h1>

<h3>Prerequisites</h3>
<ol>
<li>Python</li>
<li>Numpy</li>
<li>Pandas</li>
<li>Opencv</li>
<li>Sklearn</li>
</ol>

<h3>How to start this</h3>
Just click on the test.py file and you will get a window where you will see the live stream from your web cam, then just place your face in front of you web cam. Then if you have
weared any face mask then this will say that you are 'unsave' and when you will wear a mask then you will see 'save' written on the top of your face. For exit press 'q'.

<h3>Steps of developing this project</h3>
<ol>
<li>First, you have to collect the data means you have to collect lots of images of human faces having face mask and also those faces too which are not waring any kind of mask. In the file ml_model I hace saved a file named as dataset2.ipynb wether I have demostrated that how to fetch images from local system and how to create a proper csv file(data_2) where I have saved the cleaned datapoints.</li>
<li>Then you have clean the data and some how reach to the training and test set. In the file ml_model you will get a notebook named as training.ipynb there I have demostrated that how you can train a model.</li>
<li>Once you are done with this, then jump to sklearn and find out the best model and let me tell you that here I have used SVM model for this project</li>
<li>After that use the opencv to get the live stream from your web cam and use your model to detect the face mask.</li>
</ol>
