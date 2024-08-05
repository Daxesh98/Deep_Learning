# import kmeans model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def Deep_Learn(Y, X, X_scaled):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    tf.keras.utils.set_random_seed(42)
    
    # set model_1 to None
    model_1 = None

    # 1. Create the model using the Sequential API
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),  # Inner layer
        tf.keras.layers.Dense(1)   # Output layer
    ])
    
    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  # Binary since we are working with 2 classes (0 & 1)
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['accuracy'])

    
    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=5, verbose=0)
    evaluation = model_1.evaluate(x_train, y_train)

    return evaluation

def Model_MN(Y, X, X_scaled):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    tf.keras.utils.set_random_seed(42)

    # Set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),  # Inner Layer
        tf.keras.layers.Dense(1)   # Output layer
    ])
        # Compile the model
    model_1.compile(loss="binary_crossentropy",  # We can use strings here too
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["accuracy"])

    # Create a learning rate scheduler callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )

    # Fit the model (passing lr_scheduler callback)
    history = model_1.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[lr_scheduler])
    return history
    
def activation_function(Y, X, X_scaled):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    tf.keras.utils.set_random_seed(42)

    # Set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),  # Try activations LeakyReLU, sigmoid, Relu, tanh. Default is Linear
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    history1 = model_1.fit(x_train, y_train, epochs=50, verbose=0)
    model_1.evaluate(x_train, y_train)
    y_preds = tf.round(model_1.predict(x_test))
    
    return y_test, y_preds, history1

