import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam 
import streamlit as st

# Load and preprocess the dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize images to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
image_shape = X_train.shape[1:]

# Define the generator model
def build_generator(latent_dim):
    generator = Sequential(name="generator")
    generator.add(Dense(256, input_dim=latent_dim, name="gen_dense1"))
    generator.add(LeakyReLU(alpha=0.2, name="gen_leakyrelu1"))
    generator.add(BatchNormalization(name="gen_batchnorm1"))
    generator.add(Dense(512, name="gen_dense2"))
    generator.add(LeakyReLU(alpha=0.2, name="gen_leakyrelu2"))
    generator.add(BatchNormalization(name="gen_batchnorm2"))
    generator.add(Dense(1024, name="gen_dense3"))
    generator.add(LeakyReLU(alpha=0.2, name="gen_leakyrelu3"))
    generator.add(BatchNormalization(name="gen_batchnorm3"))
    generator.add(Dense(np.prod(image_shape), activation='tanh', name="gen_output"))
    generator.add(Reshape(image_shape))
    return generator

# Define the discriminator model
def build_discriminator(input_shape):
    discriminator = Sequential(name="discriminator")
    discriminator.add(Flatten(input_shape=input_shape, name="dis_flatten"))
    discriminator.add(Dense(512, name="dis_dense1"))
    discriminator.add(LeakyReLU(alpha=0.2, name="dis_leakyrelu1"))
    discriminator.add(Dense(256, name="dis_dense2"))
    discriminator.add(LeakyReLU(alpha=0.2, name="dis_leakyrelu2"))
    discriminator.add(Dense(1, activation='sigmoid', name="dis_output"))
    return discriminator

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential(name="gan")
    gan.add(generator)
    gan.add(discriminator)
    return gan

# Function to generate images
def generate_images(generator, num_images, latent_dim):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]
    return generated_images

# Function to evaluate image clarity
def evaluate_clarity(image):
    # For simplicity, assume image clarity based on pixel variance
    return np.var(image) > 0.1

# Main Streamlit app
def main():
    st.title('Clear Image Generator')
    
    # Parameters
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(image_shape)
    gan = build_gan(generator, discriminator)
    
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    st.write("Generating clear images...")
    clear_image_generated = False
    while not clear_image_generated:
        generated_images = generate_images(generator, num_images=1, latent_dim=latent_dim)
        clear_image_generated = evaluate_clarity(generated_images[0])
        st.image(generated_images[0].reshape(image_shape), use_column_width=True, caption='Generated Image')
        st.write(f"Image clarity: {'Clear' if clear_image_generated else 'Blurry'}. Generating next image...")

    st.write("Clear image generated!")

if __name__ == '__main__':
    main()
