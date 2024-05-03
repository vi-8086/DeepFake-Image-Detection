{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9952105a",
   "metadata": {
    "papermill": {
     "duration": 0.002323,
     "end_time": "2024-05-03T15:43:21.085458",
     "exception": false,
     "start_time": "2024-05-03T15:43:21.083135",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Loading and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f6ce8b77",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-05-03T15:43:21.091183Z",
     "iopub.status.busy": "2024-05-03T15:43:21.090354Z",
     "iopub.status.idle": "2024-05-03T15:46:12.783710Z",
     "shell.execute_reply": "2024-05-03T15:46:12.782697Z"
    },
    "papermill": {
     "duration": 171.6987,
     "end_time": "2024-05-03T15:46:12.786225",
     "exception": false,
     "start_time": "2024-05-03T15:43:21.087525",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-05-03 15:43:22.819815: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "2024-05-03 15:43:22.819915: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "2024-05-03 15:43:22.955034: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 140002 images belonging to 2 classes.\n",
      "Found 39428 images belonging to 2 classes.\n",
      "Found 190335 images belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "# Define the paths to your dataset directories\n",
    "train_dir = \"/kaggle/input/deepfake-and-real-images/Dataset/Train\"\n",
    "validation_dir = \"/kaggle/input/deepfake-and-real-images/Dataset/Validation\"\n",
    "test_dir = \"/kaggle/input/deepfake-and-real-images/Dataset\"\n",
    "\n",
    "# Define image data generators for training, validation, and test sets\n",
    "train_datagen = ImageDataGenerator(rescale=1./255)\n",
    "validation_datagen = ImageDataGenerator(rescale=1./255)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "# Set batch size\n",
    "batch_size = 64\n",
    "\n",
    "# Load and preprocess the training dataset\n",
    "train_data = train_datagen.flow_from_directory(\n",
    "    train_dir,\n",
    "    target_size=(299, 299),  # Xception requires input size of 299x299 , efficientnet- 260x260\n",
    "    batch_size=batch_size,\n",
    "    class_mode='binary'  # Set class_mode according to your dataset\n",
    ")\n",
    "\n",
    "# Load and preprocess the validation dataset\n",
    "validation_data = validation_datagen.flow_from_directory(\n",
    "    validation_dir,\n",
    "    target_size=(299, 299),\n",
    "    batch_size=batch_size,\n",
    "    class_mode='binary'\n",
    ")\n",
    "\n",
    "# Load and preprocess the test dataset\n",
    "test_data = test_datagen.flow_from_directory(\n",
    "    test_dir,\n",
    "    target_size=(299, 299),\n",
    "    batch_size=batch_size,\n",
    "    class_mode='binary'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f19149d3",
   "metadata": {
    "papermill": {
     "duration": 0.002049,
     "end_time": "2024-05-03T15:46:12.790779",
     "exception": false,
     "start_time": "2024-05-03T15:46:12.788730",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Xception"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bc22bcfc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-05-03T15:46:12.796820Z",
     "iopub.status.busy": "2024-05-03T15:46:12.796133Z",
     "iopub.status.idle": "2024-05-03T18:06:29.104208Z",
     "shell.execute_reply": "2024-05-03T18:06:29.103232Z"
    },
    "papermill": {
     "duration": 8416.313436,
     "end_time": "2024-05-03T18:06:29.106355",
     "exception": false,
     "start_time": "2024-05-03T15:46:12.792919",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
      "\u001b[1m83683744/83683744\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\n",
      "Epoch 1/10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:120: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "I0000 00:00:1714751211.189766      87 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1309s\u001b[0m 584ms/step - accuracy: 0.7338 - loss: 0.5285 - val_accuracy: 0.7705 - val_loss: 0.4772\n",
      "Epoch 2/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m707s\u001b[0m 323ms/step - accuracy: 0.7805 - loss: 0.4565 - val_accuracy: 0.7763 - val_loss: 0.4637\n",
      "Epoch 3/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m696s\u001b[0m 317ms/step - accuracy: 0.7874 - loss: 0.4452 - val_accuracy: 0.7576 - val_loss: 0.4916\n",
      "Epoch 4/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m679s\u001b[0m 309ms/step - accuracy: 0.7924 - loss: 0.4388 - val_accuracy: 0.7699 - val_loss: 0.4737\n",
      "Epoch 5/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m687s\u001b[0m 313ms/step - accuracy: 0.7925 - loss: 0.4345 - val_accuracy: 0.7827 - val_loss: 0.4563\n",
      "Epoch 6/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m698s\u001b[0m 318ms/step - accuracy: 0.7968 - loss: 0.4293 - val_accuracy: 0.7736 - val_loss: 0.4697\n",
      "Epoch 7/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m687s\u001b[0m 313ms/step - accuracy: 0.7986 - loss: 0.4288 - val_accuracy: 0.7746 - val_loss: 0.4666\n",
      "Epoch 8/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m683s\u001b[0m 311ms/step - accuracy: 0.7976 - loss: 0.4298 - val_accuracy: 0.7842 - val_loss: 0.4540\n",
      "Epoch 9/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m684s\u001b[0m 312ms/step - accuracy: 0.7987 - loss: 0.4260 - val_accuracy: 0.7795 - val_loss: 0.4605\n",
      "Epoch 10/10\n",
      "\u001b[1m2188/2188\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m705s\u001b[0m 321ms/step - accuracy: 0.8002 - loss: 0.4249 - val_accuracy: 0.7828 - val_loss: 0.4571\n",
      "\u001b[1m2974/2974\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m874s\u001b[0m 294ms/step - accuracy: 0.4151 - loss: 1.2168\n",
      "Test Loss: 1.2125420570373535, Test Accuracy: 0.41596657037734985\n"
     ]
    }
   ],
   "source": [
    "import keras\n",
    "from keras.applications import Xception\n",
    "from keras.optimizers import Adam\n",
    "import torch\n",
    "\n",
    "# Load the pre-trained Xception model without the top layers\n",
    "base_model = Xception(weights='imagenet', input_shape=(299, 299, 3), include_top=False)\n",
    "\n",
    "# Freeze the layers of the pre-trained model\n",
    "for layer in base_model.layers:\n",
    "    layer.trainable = False\n",
    "\n",
    "# Add a new output layer for binary classification\n",
    "x = base_model.output\n",
    "x = keras.layers.GlobalAveragePooling2D()(x)\n",
    "outputs = keras.layers.Dense(1, activation='sigmoid')(x)  # 2 classes for binary classification\n",
    "\n",
    "# Create the fine-tuned model\n",
    "model = keras.models.Model(inputs=base_model.input, outputs=outputs)\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "with tf.device('/GPU:0'):\n",
    "    history = model.fit(train_data, epochs=10, validation_data=validation_data)\n",
    "\n",
    "# Evaluate the model on the test set\n",
    "with tf.device('/GPU:0'):\n",
    "    loss, accuracy = model.evaluate(test_data)\n",
    "    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "gpu",
   "dataSources": [
    {
     "datasetId": 1909705,
     "sourceId": 3134515,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30699,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 8596.439452,
   "end_time": "2024-05-03T18:06:34.824177",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-05-03T15:43:18.384725",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
