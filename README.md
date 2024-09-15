# Rope Skipping Exercise Project

## Files/Scripts

- **mediapipe.ipynb**  
  *Description*: Contains the script for pose estimation using the Mediapipe library.

- **keypoints.ipynb**  
  *Description*: Extracts keypoint data from Mediapipe for further processing.

- **video_to_image.ipynb**  
  *Description*: Script used to break videos into frames to create the dataset.

- **model_training.ipynb**  
  *Description*: Contains the script for data preparation, defining the ResNet-34 model architecture, orchestrating the training process with customizable parameters, evaluating model performance, and saving model weights for future use.

- **inference.ipynb**  
  *Description*: Script for running inference on new data using the trained model.

- **Integrated.ipynb**  
  *Description*: Integrates the trained model with Mediapipe.

- **Conditions.ipynb**  
  *Description*: Script for incorporating exercise conditions for accurate analysis.

- **final_integrated.ipynb**  
  *Description*: Final integration combining the trained model, Mediapipe, and exercise conditions for accurate analysis and inference.

## Folders

- **Inference**  
  Contains the `final_integrated.py` script and a `weights` folder, crucial for running inference using the final integrated system.

- **Dataset**  
  *Description*: Stores the dataset used for training and testing the model.

- **Runs**  
  *Description*: Contains saved model weights and checkpoints from training sessions.

- **Raw_Videos**  
  *Description*: Raw video data used for creating the dataset and testing the system.

- **Test_Videos**  
  *Description*: Additional video data specifically for testing system performance.

## Required Libraries

- **matplotlib**: For plotting graphs and visualizations.
- **numpy**: For numerical computations and array operations.
- **Pillow**: For image processing tasks.
- **tensorflow**: For machine learning and deep learning tasks, including training and running neural networks.
- **opencv-python**: For computer vision tasks such as image and video processing.
- **mediapipe**: For pose estimation and other computer vision tasks provided by the Mediapipe library.

## Inference

To run the inference code using the `final_integrated.ipynb` script and the provided weights files in the "Inference" folder, follow these steps:

1. **Mount Google Drive**  
   Ensure that your drive is mounted for access.

2. **Setup Environment**  
   Install the required libraries if not already installed.

3. **Load Model Weights**  
   Set the correct path to load the model weights from the `weights` folder.

4. **Input/Output Paths**  
   Specify the paths for the input video and the output location where you want to save the resulting video and text file.

## Model Training

To train the model using the `model_training.ipynb` notebook, follow these instructions:

1. **Load Dataset**  
   Prepare your dataset for model training. Upload or access your dataset within the script.

2. **Define Model Architecture**  
   Define the architecture of your model. Use a pre-defined architecture like ResNet-34.

3. **Compile Model**  
   Specify the loss function, optimizer, and metrics for training, and compile your model.

4. **Train Model**  
   Train your model using the prepared dataset. Adjust hyperparameters such as batch size and number of epochs as needed.

5. **Save Model Weights**  
   After training, save the trained model weights to a specified location for future use.
