from keras import *
from lightgbm import LGBMClassifier
import utils
import tensorflow as tf
from  tensorflow.keras import layers


def sparse_auto_encoder(x_train,params):
    feature_size = x_train.shape[1]
    input_img = layers.Input(shape=(feature_size,))
    layer = [256, 128, 96, 64]

    # 编码层
    encoded = layers.Dense(layer[0], activation='relu')(input_img)
    encoded = layers.Dense(layer[1], activation='relu')(encoded)
    encoded = layers.Dense(layer[2], activation='relu')(encoded)
    encoder_output = layers.Dense(layer[3], activation='relu')(encoded)
    # 解码层
    decoded = layers.Dense(layer[0], activation='relu')(encoder_output)
    decoded = layers.Dense(layer[1], activation='relu')(decoded)
    decoded = layers.Dense(layer[2], activation='relu')(decoded)
    decoded = layers.Dense(feature_size, activation='tanh')(decoded)
    # KL divergence regularization
    def kl_divergence(rho, activations):
        rho_hat = tf.reduce_mean(activations, axis=0)
        kl_div = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
        return kl_div

    # Sparse loss function
    def sparse_loss(penalty=params[1], sparsity=params[2]):
        def loss(y_true, y_pred):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            activations = tf.reduce_mean(encoder_output, axis=0)
            kl_div = tf.reduce_sum(kl_divergence(sparsity, activations))
            return params[0] * mse_loss + penalty * kl_div

        return loss
    # Autoencoder model
    autoencoder = models.Model(inputs=input_img, outputs=decoded)
    # Encoder model
    encoder = models.Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss=sparse_loss())
    # Train model

    # earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=80, mode='max', verbose=0,
    #                           restore_best_weights=True)
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, verbose=0)
    encoded_imgs = encoder.predict(x_train)
    return encoder_output, encoded_imgs


def Sparse_Autoencoder(dataset_name,params):
    data, label = utils.prepare_data(dataset_name)
    encoder, data = sparse_auto_encoder(data,params)
    return data, label


def classifier(train, train_label, test):
    lgbm = LGBMClassifier(num_leaves=31,learning_rate=0.1,max_depth=-1)
    lgbm.fit(train, train_label.ravel())
    lgbm_pred = lgbm.predict_proba(test)[:, 1]

    return lgbm_pred
