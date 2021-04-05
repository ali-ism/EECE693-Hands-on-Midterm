# EECE693-Hands-on-Midterm
Midterm coding exam for the EECE693 Neural Networks course.

# Dataset

## Dataset Context

This dataset contains images of several people involved or concerned with facing the global COVID-19 pandemic along with images of other celebrities.

The people in question are:
- Fadlo Khuri
- Hamad Hassan
- Marcel Ghanem
- Andrew M. Cuomo
- Anthony Fauci
- Tedros Adhanom
- Donald Trump
- Bill Gates
- Keanu Reeves
- Cate Blanchett
- Samuel L. Jackson


## Data Collection

The images were collected by running the name of each person as search query in Google Image Search.
All images appearing in the search were downloaded at first.
12 images per person were selected and partitioned as such: 10 for training and 2 for testing.

## Data Annotation

Each image is named by the query used to find it followed by an ordinality number.
When using the data for classification, the label for each image would be its name (with the trailing digits and ".jpg" extension removed).

## Folder Content

The data is partitioned into two folder:
- "Training": contains 110 labelled images, 10 per class.
- "Testing": contains 22 labelled images, 2 per class.

# Source

This folder contains two subfolders:
- Dataset Creation and Model Development
- Model Implementation

## Dataset Creation and Model Development
The "Dataset Creation and Model Development" folder contains the information, resources, code and explanation for the process of scraping the data,
processing it, model construction, training and evaluation. This information is presented in the form of a Juppyter notebook available in both its
native "ipynb" format which can be opened in Jupyter and in "html" format for reading which can be opened in any html editor or web browser.

## Model Implementation
The "Model Implementation" folder contains the code and resources needed to deploy this model and use it to do inference.

The "model_implementation.py" file contains the Python code that needs to be run to build the model and load its learned weights and to be able
to obtain the inference for a given image. It is recommended to simply import this file as a module to use it. To run, this module depends on the
following Python libraries being installed:
- Numpy
- Tensorflow version 2 or higher

The "example.ipynb"/"example.html" files show an example of how this model can be deployed and used to classify an image downloaded from the web.
The "ipynb" file can be opened and run in Jupyter while the "html" file is just for quick reading.

The model weights can be found at: https://drive.google.com/open?id=1OEdcpvH2hv729pupwt80qaZrUKLLxx_G
