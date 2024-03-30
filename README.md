# Brain-Tumor-Classification-using-Deep-Learning

# Process Flow :
1) Dataset : 
   Taken from kaggle : https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
   Consists of 3264 tumor MRI images data for training and testing.

3) Splitting data into categories

4) Convolution Neural Network (CNN)

5) Training

6) Testing

7) Prediction

# Description :

1. **Objective**:
   - The primary goal of this project is to **classify brain tumors** based on MRI (Magnetic Resonance Imaging) scans.

2. **Data Collection and Preprocessing**:
   - **Data Source**: The project uses a dataset of MRI images.
   - **Preprocessing Steps**:
     - Load the MRI images.
     - Normalize pixel values to a common range (e.g., 0 to 1).
     - Resize the images to a consistent size.
     - Split the dataset into training and validation sets.

3. **Model Architecture**:
   - The project employs a **Convolutional Neural Network (CNN)** for tumor classification.
   - **Architecture Components**:
     - Convolutional layers with filters to learn spatial features.
     - Pooling layers for downsampling.
     - Fully connected layers for classification.

4. **Model Training**:
   - The CNN model is trained using the training data.
   - **Loss Function**: Cross-entropy loss.
   - **Optimizer**: Adam optimizer.
   - **Training Process**:
     - Forward pass: Compute predictions.
     - Backward pass: Update weights using gradients.
     - Repeat for multiple epochs.

5. **Model Evaluation**:
   - The trained model is evaluated on the validation set.
   - Metrics such as **accuracy**, **precision**, **recall**, and **F1-score** are computed.

6. **Results and Interpretation**:
   - The project reports the model's performance metrics.
   - Interpretation of results:
     - High accuracy indicates good overall performance.
     - Precision and recall provide insights into class-specific performance.

7. **Comments and Further Steps**:
   - The project includes three comments:
     1. **Data Augmentation**: Discusses techniques to enhance the dataset.
     2. **Hyperparameter Tuning**: Suggests optimizing model hyperparameters.
     3. **Transfer Learning**: Explores using pre-trained models for feature extraction.

8. **Libraries Used**:
   - **NumPy**: For numerical computations and array manipulation.
   - **Pandas**: For data manipulation and analysis.
   - **Matplotlib** and **Seaborn**: For visualization.
   - **Scikit-learn**: For machine learning tools.
   - **TensorFlow (with Keras)**: For deep learning model creation.
   - **OpenCV**: For image preprocessing.
