
from django.conf import settings
import io
import os 
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
import joblib
from skimage.transform import resize

# Load the trained model
adaboost_model = joblib.load(os.path.join(settings.BASE_DIR, 'static', 'models', 'adaboost_model.pkl'))

# # Load the trained AdaBoost model
# adaboost_model = joblib.load('adaboost_model_0.pkl')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Retrieve input image from POST request
            img_file = request.FILES['image']
            
            # Read the image file into memory
            img_data = img_file.read()
            
            # Convert image data to array
            img = image.img_to_array(image.load_img(io.BytesIO(img_data), target_size=(224, 224)))

            # Z-score normalization
            img = (img - np.mean(img)) / np.std(img)

            # Reshape image to match model input shape
            img = np.expand_dims(img, axis=0)

            # Feature extraction using ResNet50
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            features = base_model.predict(img)

            # Reshape the feature vector to have 2 dimensions (flattening)
            features = features.reshape(features.shape[0], -1)

            # Make predictions
            predictions = adaboost_model.predict(features)

            # Prepare response
            response = {'prediction': int(predictions[0])}  # Assuming predictions are binary
            return JsonResponse(response)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)