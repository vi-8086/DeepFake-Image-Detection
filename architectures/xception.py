{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"gpu","dataSources":[{"sourceId":3134515,"sourceType":"datasetVersion","datasetId":1909705}],"dockerImageVersionId":30699,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"markdown","source":"# Data Loading and Preprocessing","metadata":{}},{"cell_type":"code","source":"import tensorflow as tf\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\n\n# Define the paths to your dataset directories\ntrain_dir = \"/kaggle/input/deepfake-and-real-images/Dataset/Train\"\nvalidation_dir = \"/kaggle/input/deepfake-and-real-images/Dataset/Validation\"\ntest_dir = \"/kaggle/input/deepfake-and-real-images/Dataset\"\n\n# Define image data generators for training, validation, and test sets\ntrain_datagen = ImageDataGenerator(rescale=1./255)\nvalidation_datagen = ImageDataGenerator(rescale=1./255)\ntest_datagen = ImageDataGenerator(rescale=1./255)\n\n# Set batch size\nbatch_size = 64\n\n# Load and preprocess the training dataset\ntrain_data = train_datagen.flow_from_directory(\n    train_dir,\n    target_size=(260, 260),  # Xception requires input size of 299x299 , efficientnet- 260x260\n    batch_size=batch_size,\n    class_mode='binary'  # Set class_mode according to your dataset\n)\n\n# Load and preprocess the validation dataset\nvalidation_data = validation_datagen.flow_from_directory(\n    validation_dir,\n    target_size=(260, 260),\n    batch_size=batch_size,\n    class_mode='binary'\n)\n\n# Load and preprocess the test dataset\ntest_data = test_datagen.flow_from_directory(\n    test_dir,\n    target_size=(260, 260),\n    batch_size=batch_size,\n    class_mode='binary'\n)","metadata":{"execution":{"iopub.status.busy":"2024-05-03T15:51:54.115734Z","iopub.execute_input":"2024-05-03T15:51:54.116612Z","iopub.status.idle":"2024-05-03T15:53:08.094164Z","shell.execute_reply.started":"2024-05-03T15:51:54.116579Z","shell.execute_reply":"2024-05-03T15:53:08.093184Z"},"trusted":true},"execution_count":1,"outputs":[{"name":"stderr","text":"2024-05-03 15:51:54.478350: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n2024-05-03 15:51:54.478404: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n2024-05-03 15:51:54.480010: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n","output_type":"stream"},{"name":"stdout","text":"Found 140002 images belonging to 2 classes.\nFound 39428 images belonging to 2 classes.\nFound 190335 images belonging to 3 classes.\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# Xception","metadata":{}},{"cell_type":"code","source":"import keras\nfrom keras.applications import Xception\nfrom keras.optimizers import Adam\nimport torch\n\n# Load the pre-trained Xception model without the top layers\nbase_model = Xception(weights='imagenet', input_shape=(299, 299, 3), include_top=False)\n\n# Freeze the layers of the pre-trained model\nfor layer in base_model.layers:\n    layer.trainable = False\n\n# Add a new output layer for binary classification\nx = base_model.output\nx = keras.layers.GlobalAveragePooling2D()(x)\noutputs = keras.layers.Dense(1, activation='sigmoid')(x)  # 2 classes for binary classification\n\n# Create the fine-tuned model\nmodel = keras.models.Model(inputs=base_model.input, outputs=outputs)\n\n# Compile the model\nmodel.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])\n\n# Train the model\nwith tf.device('/GPU:0'):\n    history = model.fit(train_data, epochs=10, validation_data=validation_data)\n\n# Evaluate the model on the test set\nwith tf.device('/GPU:0'):\n    loss, accuracy = model.evaluate(test_data)\n    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')","metadata":{"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"# EfficientNetV2M","metadata":{}},{"cell_type":"code","source":"import tensorflow.keras as keras\nfrom tensorflow.keras.applications import EfficientNetV2M\nfrom tensorflow.keras.optimizers import Adam\n\n# Step 1: Load the EfficientNetV2-M model\nmodel = EfficientNetV2M(weights='imagenet', input_shape=(260, 260, 3), include_top=False)\n\n# Step 2: Freeze the convolutional base\nmodel.trainable = False\n\n# Step 3: Add custom classification head\nx = keras.layers.GlobalAveragePooling2D()(model.output)\noutputs = keras.layers.Dense(1, activation='sigmoid')(x)\n\n# Step 4: Compile the model\nmodel = keras.models.Model(inputs=model.input, outputs=outputs)\nmodel.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])\n\n# Step 5: Train the model\nhistory = model.fit(train_data, epochs=10, validation_data=validation_data)\n\n# Step 6: Evaluate the model\nloss, accuracy = model.evaluate(test_data)\nprint(f'Test Loss: {loss}, Test Accuracy: {accuracy}')\n","metadata":{},"execution_count":null,"outputs":[]}]}