import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Dropout, Softmax, Flatten, Activation
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class CovidFaces:
    def __init__(self, path_to_model_weights):
        """Instantiates the model object.
        The path to the model weights files obtained from the Google Drive link accompanying this document
        should be passed as an argument in string format."""
        vgg_face = Sequential(name='vgg_face_original')
        vgg_face.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        vgg_face.add(Convolution2D(64, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(64, (3, 3), activation='relu'))
        vgg_face.add(MaxPooling2D((2,2), strides=(2,2)))
         
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(128, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(128, (3, 3), activation='relu'))
        vgg_face.add(MaxPooling2D((2,2), strides=(2,2)))
         
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(256, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(256, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(256, (3, 3), activation='relu'))
        vgg_face.add(MaxPooling2D((2,2), strides=(2,2)))
         
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(MaxPooling2D((2,2), strides=(2,2)))
         
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(ZeroPadding2D((1,1)))
        vgg_face.add(Convolution2D(512, (3, 3), activation='relu'))
        vgg_face.add(MaxPooling2D((2,2), strides=(2,2)))
         
        vgg_face.add(Convolution2D(4096, (7, 7), activation='relu'))
        vgg_face.add(Dropout(0.5))
        vgg_face.add(Convolution2D(4096, (1, 1), activation='relu'))
        vgg_face.add(Dropout(0.5))
        vgg_face.add(Convolution2D(2622, (1, 1)))
        vgg_face.add(Flatten())
        vgg_face.add(Activation('softmax'))

        vgg_base = Model(inputs=vgg_face.layers[0].input, outputs=vgg_face.layers[-2].output, name='vgg_base')

        fc = Dense(1024, activation='relu')(vgg_base.layers[-1].output)
        drp = Dropout(0.5)(fc)
        fc_out = Dense(11, activation='softmax')(drp)

        self.model = Model(inputs=vgg_base.input, outputs=fc_out, name='final_model')

        #load model weights
        self.model.load_weights(path_to_model_weights)
        
    def model_summary(self):
        self.model.summary()
        
    def _preprocess_image(self, path_to_image):
        img = load_img(path_to_image, color_mode='rgb', target_size=(224,224))
        img = img_to_array(img)
        img = img / 255
        img = img[np.newaxis]
        return img
    
    def infer_image(self, path_to_image):
        """This function expects the path to the image to be classified as an argument in string format.
        A Tensorflow model instance must also be passed as an argument.
        The image is the processed into the proper format and fed to the model for inference."""
        img = self._preprocess_image(path_to_image)
        #infer image class
        pred = np.argmax(self.model.predict(img, batch_size=1), axis=1)
        #map numerical prediction to class name
        class_names = ['Andrew M. Cuomo', 'Anthony Fauci', 'Bill Gates', 'Cate Blanchett', 'Donald Trump', 'Fadlo Khuri', 'Hamad Hassan', 'Keanu Reeves',
        'Marcel Ghanem', 'Samuel L. Jackson', 'Tedros Adhanom']
        return class_names[pred[0]]