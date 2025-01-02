# Breast Cancer Detection MLOps

| **Category**          | **Description**                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------|
| **Dataset**           | The dataset contains radiological numerical features of breast cancer patients, such as `radius_mean`, `texture_mean`, `perimeter_mean`, and more. <br> Dataset: [Wisconsin Diagnostic Breast Cancer (WDBC)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). |
| **Problem**           | Early detection of breast cancer is crucial for improving patient recovery rates. <br> Machine learning-based classification can assist in diagnosing cancer more quickly and accurately. |
| **Machine Learning Solution** | A **classification model** that predicts whether the data belongs to the **Malignant (cancer)** or **Benign (non-cancer)** category using numerical features. |
| **Data Processing Methods** | The data is preprocessed using: <br> - **Normalization** of numerical features to bring them to the same scale. <br> - **Splitting** the dataset into training, validation, and testing sets. |
| **Model Architecture** | The model is built using a **Dense Neural Network (DNN)**: <br> - **Input Layer**: Numerical features such as `radius_mean`, `texture_mean`, `perimeter_mean`, etc. <br> - **Hidden Layers**: 2-3 dense layers with ReLU activation. <br> - **Output Layer**: Sigmoid activation for binary classification. |
| **Evaluation Metrics** | The model is evaluated using: <br> - **Binary Accuracy** <br> - **Precision** <br> - **Recall** <br> - **F1-Score** |
| **Model Performance** | The model achieves an accuracy of **95% or higher** on the test data, with a good balance between **Precision** and **Recall**. |
| **Deployment Option** | The model can be deployed on cloud platforms such as **AWS SageMaker**, exposing an API endpoint for integration with clinical applications. |
| **Web App**            | A simple web application allows users to upload patient data and receive predictions.                |
| **Monitoring**         | Monitoring is implemented using **Prometheus** and **Grafana** to track the number of predictions, response times, and request statuses (successful/failed). |
