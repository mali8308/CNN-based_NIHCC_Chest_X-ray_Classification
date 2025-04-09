#!/usr/bin/env python
# coding: utf-8

# # Homework 2 - Convolutional Neural Networks
# 
# ### Deep Learning in Medicine - Spring 2025

# 
# 
# **Note:** If you need to write mathematical terms, you can type your answeres in a Markdown Cell via LaTex
# 
# **See:** <a href="https://stackoverflow.com/questions/13208286/how-to-write-latex-in-ipython-notebook">here</a> if you have issues. To see basic LaTex notation see: <a href="https://en.wikibooks.org/wiki/LaTeX/Mathematics"> here </a>.
# 
# **Submission instruction:** Upload and Submit a zipped folder named netid_hw2 consisting of your final jupyter notebook and necessary files in <a href='https://brightspace.nyu.edu/d2l/home/427921'>Brightspace</a>. If you use code or script from web, please give a link to the code in your answers. Not providing the reference of the code used will reduce your points!!
# 
# **Submission deadline: Saturday March 20rd, 2025**

# ### Topics & weightage -
# 
# 
# 1.   Convolutions (30)
# 2.   Network design (15)
# 3.   Literature review (19)
# 4.   Deep CNN design for disease classification (36)
# 5.   Analysis of Results (5)
# 6.   Bonus Questions (12) - optional!
# 
# 

# ## Question 1 Convolutions (Total 30 points)

# ### 1.1 Convolutions from **scratch** for image processing (11 points)

# In[2]:


import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# In[6]:


# functions to plot images
def plot_image(img: np.array):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray');
    
def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')


# #### 1.1.a (1 point)

# In[4]:


# TODO: load any image of your choice and display (plot) the resized image (224*224) in grayscale using the plot_image function
# or you can also utilize the sample image provided --> cat.png
# (none of these transformations are mandatory, but they make our job a bit easier, 
# as there’s only one color channel to apply convolution to)


# In[ ]:


# Load the image (replace 'cat.png' with your image file if using a different one)
image_path = "cat.png"  # Change this to the path of your image if needed
image = Image.open(image_path)

# Convert to grayscale
image_gray = ImageOps.grayscale(image)

# Resize to 224x224
image_resized = image_gray.resize((224, 224))

# Convert to numpy array
image_array = np.array(image_resized)

# Display the image
plot_image(image_array)


# In[10]:


# defining filters 
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125],
    [0.0625, 0.125, 0.0625]
])

outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


# #### 1.1.b (1.5 points)

# In[92]:


def calculate_target_size(img_size: int, kernel_size: int) -> tuple:
  
  size = img_size - kernel_size + 1

  return size, size

print("The dimensions of the picture after convolution:", calculate_target_size(224, 3))  # Expected output: (222, 222)


# #### 1.1.c (3 points)

# In[51]:


def convolve(img: np.array, kernel: np.array) -> np.array:

    size_of_target = calculate_target_size(img.shape[0], kernel.shape[0])
    matrix_of_zeroes = np.zeros(size_of_target)

    for i in range(size_of_target[0]):
        for j in range(size_of_target[0]):
            # Extract the region of the image that matches the kernel size
            img_patch = img[i:i+kernel.shape[0], j:j+kernel.shape[0]]
            
            # Perform element-wise multiplication and sum the result
            convolved_value = np.sum(img_patch * kernel)
            
            # Store the computed value in the output image
            matrix_of_zeroes[i, j] = convolved_value

    return matrix_of_zeroes


# #### 1.1.d (0.5 point)

# In[8]:


# TODO: use the convolved function & the sharpen filter to obtain a sharpened image of your original input 
# TODO: print the sharpened image array named img_sharpened
# TODO: use the plot_two_images function to plot the original image and sharpened image side by side


# In[35]:


img_sharpened = convolve(image_array, sharpen)
print("Sharpened image array:\n", img_sharpened)

plot_two_images(image_array, img_sharpened)


# #### 1.1.e (0.5 point)

# In[53]:


def negative_to_zero(img: np.array) -> np.array:
  
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i, j] < 0:
        img[i, j] = 0
  
  return img  
  '''
  Args:
    img: numpy array of image
  
  Returns:
    img: all values less than zero are assigned zero in original image
  '''
  # TODO: the sharpened image is a little dull, thats because some values in the sharpened image 
  # are less than zero
  # write a function that uses 0 as a threshold and converts all pixel values less than zero to zero

sharpened_zeroed = negative_to_zero(img_sharpened)  
plot_two_images(image_array, sharpened_zeroed)

# TODO: use the plot_two_images function to plot the original image and negative_to_zero sharpened image side by side


# #### 1.1.f (1 point)

# In[10]:


# TODO: use the convolved function & the blur filter to obtain a blurred image of your original input 
# TODO: print the blurred image array named img_blurred
# TODO: use the plot_two_images function to plot the original image and blurred image side by side


# In[56]:


img_blurred = convolve(image_array, blur)
print("Blurred image array:\n", img_blurred)

plot_two_images(image_array, img_blurred)


# In[11]:


# TODO: use the convolved function & the outline filter to obtain a outlined image of your original input 
# TODO: print the outlined image array named img_outlined
# TODO: use the plot_two_images function to plot the outlined image and original image side by side


# In[58]:


img_outlined = convolve(image_array, outline)
print("Outlined image array:\n", img_outlined)

plot_two_images(image_array, img_outlined)


# **Reminder:** Padding is essentially a “black” border around the image. It’s black because the values are zeros, and zeros represent the color black. The black borders don’t have any side effects on the calculations, as it’s just a multiplication with zero.

# #### 1.1.g (0.5 point)

# In[59]:


def get_padding_width_per_side(kernel_size: int) -> int:
    '''
    Function that returns the number of pixels we need to 
    pad the image with on a single side, depending on the kernel size

    Args:
    kernel_size: filter size 

    Returns:
    padding_width 
    '''
    # TODO: simple integer division by 2

    padding_width = kernel_size // 2

    return padding_width


# In[60]:


pad_3x3 = get_padding_width_per_side(3)
pad_5x5 = get_padding_width_per_side(5)
print("padding for kernel size 3 is", pad_3x3, "and padding for kernel size 5 is", pad_5x5)


# #### 1.1.h (1.5 points)

# In[80]:


def add_padding_to_image(img: np.array, padding_width: int) -> np.array:
    
    matrix_of_zeroes = np.zeros((img.shape[0] + padding_width*2, img.shape[1] + padding_width*2))

    matrix_of_zeroes[padding_width:matrix_of_zeroes.shape[0]-padding_width,
                     padding_width:matrix_of_zeroes.shape[1]-padding_width] = img

    img_with_padding = matrix_of_zeroes
    
    '''
    Function that adds padding to the image. 
    First, the function declares a matrix of zeros with a shape of image.shape + padding * 2. 
    The function then indexes the matrix so the padding is ignored and changes the zeros with the actual image values.

    Args:
      img: Original image numpy array
      padding_width: obtained in the get padding function earlier

    Returns:
      img_with_padding: padded image
    '''
    # TODO: take your image and a padding width as input and return the image with the padding added
    return img_with_padding

print("Original image with padding needed for a kernel of size 3:\n", 
      add_padding_to_image(image_array, get_padding_width_per_side(3)))


# #### 1.1.i (1 point)
# 
# In the above function add_padding_to_image, explore the possible reason for the multiplication of padding_width by 2 in step 1

# **Answer**: We did that because we had to add padding on both sides, and the padding_width only told us about the size of padding on one side. 

# #### 1.1.j (0.5 point)

# In[81]:


# TODO: use the add_padding_to_image function to obtain the padded image (kernel size of 3)
img_with_padding_3x3 = add_padding_to_image(image_array, get_padding_width_per_side(3))

print(img_with_padding_3x3.shape)
plot_image(img_with_padding_3x3)


# In[82]:


# TODO: use the add_padding_to_image function to obtain the padded image (kernel size of 5)
img_with_padding_5x5 = add_padding_to_image(image_array, get_padding_width_per_side(5))

print(img_with_padding_5x5.shape)
plot_image(img_with_padding_5x5)


# #### 1.1.k (1 point)

# In[17]:


# TODO: use the convolved function & the sharpen filter and negative to zero to obtain a sharpened image of your
# padded image (kernel size of 5) obtained from add_padding_to_image function 
# TODO: print the shape of the obtain sharpened image (obtained after padding)
# TODO: plot the original image and the sharpened image (obtained after padding) side by side using the
# plot_two_images function


# In[87]:


sharpened_padded_zeroed_image = negative_to_zero(convolve(img_with_padding_5x5, sharpen))
print("Shape of the sharpened image after padding:", sharpened_padded_zeroed_image.shape)

plot_two_images(image_array, sharpened_padded_zeroed_image)


# ### 1.2 Convolutional Layers (4 points)
# 
# We have a 3x5x5 image (3 channels) and three 3x3x3 convolution kernels as pictured. Bias term for each feature map is also provided. For the questions below, please provide the feature/activation maps requested, please provide the python code that you used to calculate the maps.
# 
# **Hint:** An image tensor should be [batch size, channels, height, weight], kernels/filters tensor should be [number of filters (output channels), filter_size_1 (input channels), filter_size_2, filter_size_3].
# 
# <img src="https://github.com/nyumc-dl/BMSC-GA-4493-Spring2022/blob/main/Homework2/HW2_picture1.png?raw=1">

# What will be the dimension of the feature maps after we forward propogate the image using the given convolution kernels for the following (a) - (d)

# #### 1.2.a stride=1, padding = 0 (1 point)

# In[107]:


import math

def calculate_feature_map_dimensions(img_size: int, kernel_size: int, depth: int, stride: int) -> tuple:
  
  output_size = math.floor((img_size - kernel_size) / stride) + 1  # Ensure correct rounding

  return depth, output_size, output_size

print("The dimensions of the feature maps after forward propagation will be:", calculate_feature_map_dimensions(img_size = 5, 
                                                                                kernel_size = 3,
                                                                                depth = 3,
                                                                                stride = 1))  


# #### 1.2.b stride=2, padding = 1 (1 point) 

# In[121]:


dummy_image = (np.zeros([5,5]) + 1)

dummy_image_with_padding = (add_padding_to_image(dummy_image, 1))

print("The dimensions of the feature maps after forward propagation will be:", calculate_feature_map_dimensions(img_size = dummy_image_with_padding.shape[0], 
                                                                                kernel_size = 3,
                                                                                depth = 3,
                                                                                stride = 2))  


# #### 1.2.c stride=3, padding = 2 (1 point)

# In[122]:


dummy_image_with_padding = (add_padding_to_image(dummy_image, 2))

print("The dimensions of the feature maps after forward propagation will be:", calculate_feature_map_dimensions(img_size = dummy_image_with_padding.shape[0], 
                                                                                kernel_size = 3,
                                                                                depth = 3,
                                                                                stride = 3))  


# #### 1.2.d stride=1, dilation rate=2, and padding=0 (1 point) 

# In[127]:


def image_dimensions_with_dilated_kernel(img_size: int, kernel_size: int, dilation: int, stride: int, depth = int):
  
    dilated_kernel_size = (kernel_size - 1) * dilation + 1

    output_size = math.floor((img_size - dilated_kernel_size) / stride) + 1
    
    return depth, output_size, output_size

print("The dimensions of the feature maps after forward propagation will be:", image_dimensions_with_dilated_kernel(img_size = dummy_image.shape[0], 
                                                                                kernel_size = 3,
                                                                                depth = 3,
                                                                                stride = 1,
                                                                                dilation = 2))  


# ### 1.3 Feature Dimensions of Convolutional Neural Network (4*0.5 points)
# 
# In this problem, we compute output feature shape of convolutional layers and pooling layers, which are building blocks of CNN. Let’s assume that input feature shape is C x W × H, where C is the number of channels, W is the width, and H is the height of input feature. 
# 
# 

# 
# #### 1.3.a (0.5 points)
# 
# A convolutional layer has 4 hyperparameters: the filter size(K), the padding size (P), the stride step size (S) and the number of filters (F). How many weights and biases are in this convolutional layer? And what is the shape of output feature that this convolutional layer produces?

# Number of kernels we use $ = F$
# 
# Weights per filters $= C \times K \times K$
# 
# Total number of weights $= F \times C \times K \times K$
# 
# Total number of biases $= F$
# 
# **Shape of the output**:
# 
# $C, {\frac{(W + P) - K}{S}}+1, {\frac{(H + P) - K}{S}}+1$
# 
# 
# 

# 
# #### 1.3.b (0.5 points)
# 
# A pooling layer has 2 hyperparameters: the stride step size(S) and the filter size (K). What is the output feature shape that this pooling layer produces?

# **From the last part**:
# 
# $Width_{Convolve} =  {\frac{(W + P) - K}{S}}+1$
# 
# $Height_{Convolve} =  {\frac{(H + P) - K}{S}}+1$
# 
# **Shape of the output**:
# 
# $C, {\frac{Width_{Convolve} - K}{S}}+1, {\frac{Height_{Convolve} - K}{S}}+1$

# 
# #### 1.3.c (0.5 points)
# 
# Let’s assume that we have the CNN model which consists of L successive convolutional layers and the filter size is K and the stride step size is 1 for every convolutional layer. Then what is the receptive field size?

# Receptive Field $= K + (L-1) \times (K-1)$
# 
# 
# 
# 
# 

# 
# #### 1.3.d (0.5 points)
# 
# Consider a downsampling layer (e.g. pooling layer and strided convolution layer). In this problem, we investigate pros and cons of downsampling layer. This layer reduces the output feature resolution and this implies that the output features loose the certain amount of spatial information. Therefore when we design CNN, we usually increase the channel length to compensate this loss. For example, if we apply the max pooling layer with kernel size of 2 and stride size of 2, we increase the output feature size by a factor of 2. If we apply this max pooling layer, how much the receptive field increases? Explain the advantage of decreasing the output feature resolution with the perspective of reducing the amount of computation.

# **Answer**: The receptive field doubles in size after each downsampling step. The advantage of using downsampling layers is that they reduce the size of the input feature maps to the next layer, while still preserving important information about the original image's features. Therefore, the smaller size of the input allows for faster computations. 

# In[4]:


import torch
import torch.nn.functional as F

#Dummy image
x = torch.tensor([
    [[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]
], dtype=torch.float32)

print("Original feature map:")
print(x.squeeze())  # Remove batch and channel dimension for readability

# Apply Max Pooling with kernel_size=2, stride=2
pooled_x = F.max_pool2d(x, kernel_size=2, stride=2)

print("\nPooled feature map:")
print(pooled_x.squeeze())

# Checking receptive field growth
receptive_field_size_before = 1  # Each pixel originally sees only itself
receptive_field_size_after = receptive_field_size_before * 2  # Doubles after pooling

print(f"\nReceptive field before pooling: {receptive_field_size_before}x{receptive_field_size_before}")
print(f"Receptive field after pooling: {receptive_field_size_after}x{receptive_field_size_after}")


# ### 1.4 (6 points)
# Use the pytorch package to calculate feature/activation maps. Write a code which takes 3x5x5 image and performs a 2D convolution operation (with stride = 1 and zero padding) using 3x3x3 filters provided on the picture. After convolution layer use leaky ReLU activation function (with negative slope 0.01) and Max-Pooling operation with required parameters to finally obtain output of dimension 3x1x1. Provide the code, feature maps obtained from convolution operation, activation maps, and feature maps after Max-Pooling operation.
# 
# **Hint:** You can refer to [AdaptiveMaxPool2d](https://pytorch.org/docs/stable/nn.html#adaptivemaxpool2d) to get desired dimension output from Pooling layer.

# In[4]:


# starter code to load image:x, kernel weights:w and bias:b
# if you hit errors related to the long data type convert the values in your numpy arrays to floats
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

x = np.load('q1_input.npy')
w = np.load('q1_Filters.npy')
b = np.load('q1_biases.npy')

# Making sure the shapes of the input and the filter match. 
print("Input image shape:", x.shape,
      "\nFilter shape:", w.shape, # Shape: (3, 3, 3, 3) (out_channels, in_channels, height, width)
      "\nBias shape:", b.shape)

#conv2d expects input in the form (batch_size, num_channels, height, width). However, our image has the size 3x5x5. 
#So we add another dimension for the batch using unsqueeze. 

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  
w_tensor = torch.tensor(w, dtype=torch.float32)    
b_tensor = torch.tensor(b, dtype=torch.float32)  

print("\nInput tensor shape:", x_tensor.shape,
      "\nFilter tensor shape:", w_tensor.shape,
      "\nBias tensor shape:", b_tensor.shape)

conv_output = F.conv2d(x_tensor, 
                       w_tensor, 
                       bias=b_tensor, 
                       stride=1, 
                       padding=0)  

print(f"\nShape after convolution", conv_output.shape)
print("\nFeature maps after convolution:\n", conv_output)

# Apply Leaky ReLU activation function
activation_output = F.leaky_relu(conv_output, 
                                 negative_slope=0.01)  # Shape: (1, 3, 3, 3)

print("\nActivation output:\n", activation_output)


max_pool = nn.AdaptiveMaxPool2d((1,1))

# Apply Adaptive Max Pooling to get (3x1x1)
pooled_output = max_pool(activation_output) 

print("\nPooled output:\n", pooled_output)



# ### 1.5 (7 points)
# Use the pytorch package to calculate feature/activation maps of a residual unit. Example of a residual unit are seen in figure 2 of https://arxiv.org/pdf/1512.03385.pdf as well as in the figure below.
# 
# 
# <img src="https://github.com/nyumc-dl/BMSC-GA-4493-Spring2022/blob/main/Homework2/HW2_picture2.png?raw=1" width="150">
# 
# Write a code which takes 3x5x5 input image and performs two 2D convolution operations using the filters provided in the figure above. Please use the three 3x3x3 filters for the two Convolution layers. You need to set a suitable padding size for the convolution operations. After the convolution layers have the residual addition and use the ReLU activation function. Provide the code and feature maps obtained from each convolution operation, activation maps, and the last activation map obtained from the residual unit.

# **Residual Unit** = $F(x) + x$ where $x$ is the input to the neural network, and $F(x)$ represents all the layers between the input and final activation function before it gets passed on to the ReLU function.

# In[33]:


import torch.nn as nn

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        
        super(ResidualBlock, self).__init__()
        self.convolution_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)        
        self.ReLU_Layer = nn.ReLU()

    def forward(self, x):
        identity = x
        feature_map_1 = self.convolution_layer(x)
        activation_map_1 = self.ReLU_Layer(feature_map_1)
        feature_map_2 = self.convolution_layer(activation_map_1)
        feature_map_3 = feature_map_2 + identity
        final_activation_map = self.ReLU_Layer(feature_map_3)
        return feature_map_1, activation_map_1, feature_map_2, final_activation_map


# In[34]:


residual_block = ResidualBlock(in_channels=x_tensor.shape[1], 
                               out_channels=x_tensor.shape[1], 
                               kernel_size=w_tensor.shape[2],
                               padding=1)

residual_block.convolution_layer.weight = nn.Parameter(w_tensor)
residual_block.convolution_layer.bias = nn.Parameter(b_tensor)


# In[41]:


with torch.no_grad():
    print(f"\nFirst feature map:\n", residual_block(x_tensor)[0],"\n",
          "\nFirst activation map:\n", residual_block(x_tensor)[1],"\n",
          "\nSecond feature map:\n", residual_block(x_tensor)[2],"\n",
          "\nFinal activation map:\n",residual_block(x_tensor)[3])


# ### 1.6 (2 points)
# Describe the key design paramters of inception v3 (https://arxiv.org/pdf/1512.00567.pdf) and explain how it avoids overfitting of data.

# **Answer**:
# 
# - **Factorized & Asymmetric Convolutions**: It uses asymmetric kernels and breaks down larger kernels into smaller ones to improve efficiency.
# 
# - **Auxilliary Classifiers**: The network has multiple classifiers along the way before reaching the final classifier. That helps the model learn when it's making incorrect predictions during the training and backpropagate to correct them. This helps in reducing overfitting because the model is penalized even before it has reached the final classifier layer. 
# 
# - **Label Smoothing**: Label smoothing helps the model in learning to generalize better becuase it reduces hard predictions down to softened probabilities.
# 
# - **Normalization**: Like any other model, normalization helps inception v3 avoid overfitting the data. 

# ## Question 2 Network design parameters for disease classification (Total 15 points)

# Disease classification is a common problem in medicine. There are many ways to solve this problem. Goal of this question is to make sure that you have a clear picture in your mind about possible techniques that you can use in such a classification task.
# 
# Assume that we have a 10K images in a dataset of computed tomography (CTs). For each image, the dimension is 16x256x256 and we have the label for each image. The label of each image defines which class the image belongs (lets assume we have 4 different disease classes in total). You will describe your approach of classifying the disease for the techniques below. Make sure you do not forget the bias term. Please provide the pytorch code which designs the network for questions 2.1.a, 2.2.a, and 2.3.a.
# 
# **Hint:** See lab 4 for an example of how to make a class for a network (Implementing LeNet).
# 

# In[2]:


import torch

# starter code
# you can generate a random image tensor for batch_size 8
x = torch.Tensor(8,1,16,256,256).normal_().type(torch.FloatTensor)


# #### 2.1.a (2 points)
# Design a multi layer perceptron (MLP) with a two hidden layer which takes an image as input (by reshaping it to a vector: let's call this a vectorized image). Our network has to first map the vectorized images to a vector of 512, then to 256 in a hidden layer and then to 128 in a hidden layer and finally feeds this vector to a fully connected layer to get the probability of 5 tissue classes. 

# In[4]:


import torch.nn as nn
import torch.nn.functional as F

class Manually_Designed_Network(nn.Module):
    def __init__(self):
        super(Manually_Designed_Network, self).__init__()
        # Define layers
        self.input_layer = nn.Linear(in_features=16*256*256, 
                                     out_features=512)  
        
        self.hidden_layer_1 = nn.Linear(in_features=512, 
                                        out_features=256)
        
        self.hidden_layer_2 = nn.Linear(in_features=256, 
                                        out_features=128)

        self.output_layer = nn.Linear(in_features=128,
                                      out_features=5)  

    def forward(self, x):
        # Define the forward pass
        x = torch.flatten(x, start_dim=1)
        vectorize_image = self.input_layer (x)
        x1 = self.hidden_layer_1(vectorize_image)
        x2 = self.hidden_layer_2(x1)
        output = self.output_layer(x2)
        log_probs = F.log_softmax(output, dim=1)
        return (torch.exp(log_probs))


# Example of how to initialize the network
manually_designed_model = Manually_Designed_Network()

manually_designed_model(x)


# #### 2.1.b (2 points)
# 
# Clearly mention the sizes for your input and output at each layer until you get final output vector with 5 tissue classes and an input of images of size 16x256x256.

# The sizes of the input and output are as follows:
# 
# 1. **Input layer**: input size is 1,048,576 and the output size is 512
# 2. **First hidden layer**: input size is 512 and the output size is 256
# 3. **Second hidden layer**: input size is 256 and the output size is 128
# 4. **Third hidden layer**: input size is 128 and the output size is 5

# #### 2.1.c (1 points)
# How many parameters you need to fit for your design? How does adding another hidden layer (map to 64 after 128) will effect the number of parameters to use?

# In[5]:


for name, param in manually_designed_model.named_parameters():
    print(f"Layer: {name}, Parameters: {param.numel()}\n")

#If we add another layer to the network, the number of parameters will increase. Below I am creating a new version of the model with the aforementioned layer. 

import torch.nn as nn

class One_More_Layer(nn.Module):
    def __init__(self):
        super(One_More_Layer, self).__init__()
        # Define layers
        self.input_layer = nn.Linear(in_features=16*256*256, 
                                     out_features=512)  
        
        self.hidden_layer_1 = nn.Linear(in_features=512, 
                                        out_features=256)
        
        self.hidden_layer_2 = nn.Linear(in_features=256, 
                                        out_features=128)
        
        self.hidden_layer_2 = nn.Linear(in_features=128, 
                                        out_features=64)


        self.output_layer = nn.Linear(in_features=64,
                                      out_features=5)  

    def forward(self, x):
        # Define the forward pass
        x = torch.flatten(x, start_dim=1)
        vectorize_image = self.input_layer (x)
        x1 = self.hidden_layer_1(vectorize_image)
        x2 = self.hidden_layer_2(x1)
        x3 = self.hidden_layer_3(x2)
        output = self.output_layer(x3)
        log_probs = F.log_softmax(output, dim=1)
        return (torch.exp(log_probs))

one_more_layer = One_More_Layer()

for name, param in one_more_layer.named_parameters():
    print(f"Model with an additional layer:\nLayer: {name}, Parameters: {param.numel()}")


# #### 2.2.a (2 points)
# Design a one layer convolutional neural network which first maps the images to a vector of 256 and then 128 (both with the help of convolution and pooling operations) then feeds this vector to a fully connected layer to get the probability of 5 disease classes.

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseCNN(nn.Module):
    def __init__(self):
        super(DiseaseCNN, self).__init__()
        
        # First convolutional layer with 1 input channel (grayscale), and 16 output channels. 
        # The greater number of output channels help in enhancing the feature extraction of the model. 

        # The 16 here is basically the number of kernels we are using - since each kernel can learn different features, using more kernels means that we can learn more features.
        # For instance, there's one kernel to learn horizontal edges, one to learn vertical edges, one for textures and so on. 
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        
        # Second convolutional layer with 16 input channels, 32 output channels.
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)

        #Downsampling layer
        self.downsapling = nn.MaxPool3d(kernel_size=2, stride=2) 
        
        # Fully connected layer to output 5 disease class probabilities
        self.fc1 = nn.Linear(in_features= 32*4*64*64, #Because each downsampling layer is halving the dimensions. So the first layer halves 16 and 256. The second halves 8 and 128.
                             out_features = 128)  
        
        self.fc2 = nn.Linear(in_features = 128, 
                             out_features = 5)  # 5 classes output

    def forward(self, x):
        conv_1 = self.conv1(x)
        pool_1 = self.downsapling(conv_1)
        conv_2 = self.conv2(pool_1)
        pool_2 = self.downsapling(conv_2)
        x_flattened = torch.flatten(pool_2, 1)
        fc1_out = self.fc1(x_flattened)
        fc2_out = self.fc2(fc1_out)

        log_probs = F.log_softmax(fc2_out, dim=1)
        return (torch.exp(log_probs))

Disease_model = DiseaseCNN()

print(Disease_model(x))


# ### 2.2.b (2 points)
# Clearly mention the sizes for your input, kernel, pooling, and output at each step until you get final output vector with 5 probabilities.

# **Size of the Input**: (8,1,16,256,256) - Here the batch size is 8, there's only 1 channel (grayscale), and then there's a 3D image with 16 depth, and 256 $\times$ 256 spatial size.
# 
# **Size After the First Covolution Layer**: (8,16,16,256,256) - The size remains the same because we are using padding of 1, with a stride of 1, and kernel size of 1. At this step, the number of output channels increase to 16.
# 
# **Size After the First Pooling Layer**: (8,16,8,128,128) - Because of the kernel size 2 and the stride of 2, the downsampling layer reduces the dimensions by half. 
# 
# **Size After the Second Covolution Layer**: (8,32,8,128,128) - The size does not change after the first pooling layer because at this layer we are using padding of 1, with a stride of 1, and kernel size of 1. At this step, the number of output channels increase to 32.
# 
# **Size After the Second Pooling Layer**: (8,32,4,64,64) - Because of the kernel size 2 and the stride of 2, the downsampling layer reduces the dimensions by half. 
# 
# **Size of the Input to the First Fully Connected Layer**: (32 $\times$ 4 $\times$ 64 $\times$ 64)
# 
# **Size of the Output of the First Fully Connected Layer**: 128
# 
# **Size of the Input to the Second Fully Connected Layer**: 128
# 
# **Size of the Output of the Second Fully Connected Layer**: 5

# #### 2.2.c (1 point) 
# How many parameters you need to fit for your design?

# In[10]:


for name, param in Disease_model.named_parameters():
    print(f"\nLayer: {name}, Parameters: {param.numel()}")


# ### 2.2.d (2 points)
# Now increase your selected convolution kernel size by 4 in each direction. Describe the effect of using small vs large filter size during convolution.

# **Answer**: Larger kernel sizes allow us to look at a wider receptive field. This means that we need less layers to get the same level of receptive field as with a smaller kernel. However, we now have more parameters to compute so it might be more computationally intense. While smaller kernels look at smaller features and are *localized* so to speak, bigger kernels are more *global* in that they look at larger sections of the image and identify bigger patterns across the image.

# ### 2.3 (3 points)
# Explain your findings regading different types of neural networks and building blocks based on your observations from 2.1 and 2.2. 

# **Answer**: I would like to address three main differences that I observed: computation time, number of parameters, and architecture.
# 
# 1. **Computation Time**: At least for me, the neural network with just fully connected layers ran faster than the convolutional neural network. 
# 
# 2. **Number of Parameters**: Compared to fully connected layers, convolution layers have thousands of times smaller number of parameters.
# 
# 3. **Architecture**: For a neural network using only fully connected layers, the architecture seems to be oversimplifying the image recognition task. Especially when we move to a 3D space. On the other hand, with convolutional layers, the ability to increase output channels at will allows us to learn weights that target specific features in the image. I think this is really helpful especially because it allows us to extract meaningful features all across the image. 

# ## Question 3 Literature Review: ChestX-ray8 (Total 19 points)
# Read this paper:
# 
# Pranav Rajpurkar, Jeremy Irvin, et al. 
# CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning https://arxiv.org/abs/1711.05225
# 
# 
# We are interested in understanding the goal of the task performed, the methods proposed, technical aspects of the implementation, and possible future work. After you read the full article answer the following questions. Describe your answers in your own words.  

# ### 3.1 (2 points) 
# 
# What was the underlying goal of this paper? What were the challenges in detection of pneumonia that this paper aimed at solving? What was the key motivation?
# 

# **Answer**: The paper built a convolutional neural network with 121 layers to detect pneumonia. The challenge was that chest X-rays were the main form of diagnosis for pneumonia, however, expert practicing radiologists were required to interpret the X-rays and diagnose pneumonia. This meant that areas that didn't have access to an expert radiologist could not precisely diagnose pneumonia. And even in areas with radiologists, the X-rays were hard to diagnose because pneumonia's presentation is often vague or similar to other benign abnormalities which confuses the radiologists and they have varigying opinions about the interpretation of the X-ray. Therefore, the model tries to solve this problem by detecting pneumonia with better accuracy than radiologists. This can help improve healthcare accessibility to underserved areas and help radiologists in diagnosing the disease more efficiently.

# ### 3.2  (3 points)
# Describe the machine learning task (segmentation, classification, regression, etc?) that was attempted in this paper. Further describe the learning algorithm used (supervised, unsupervised, ..etc.) and the reason was using this algorithm.

# **Answer**: The task was binary classification (and later multi-class classification) using supervised learning. The reason they used supervised learning was because they trained the model on the ChestX-ray14 dataset, which had invidiual labels for each X-ray image the thoracic condition. They wanted to classify someone's X-ray as either having pneumonia or not, given the frontal chest X-ray image. Later on, for comparison against other models, they changed their model to predict the probability of all 14 classes of thoracic conditions present in ChestX-ray14. 

# ### 3.3 (2.5 points)
# How does the proposed architecture in this paper compare with the previous State of the art? Give details on the modifications and improvements, and reasons for why you think these worked.

# **Answer**: The ChexNet model outperforms the previous state of the art models (Yao et al., 2017 and Wang et al., 2017), especially for diagnosing Mass, Nodule, Pneumonia, and Emphysema where the ChexnNet's per-class AUROC was >0.05 higher than the older models. 
# 
# The architecture of ChexNet is different from the other models in the following aspects:
# 
# 1. **Uses 121 DenseNet architecture**: This architecture allows for better flow of gradients, which makes optimization a lot more tractable. 
# 
# 2. **Initialized pre-trained weights**: ChexNet initializes weights from a previous model trained on ImageNet, so it already started from a somewhat optimized state. And then it further optimized those weights which likely improved its performance compared to the other models.
# 
# 3. **Final layer**: The final fully connected layer is replaced by a layer that outputs a 14-dimensional vector of probabilities for each class, after having sigmoid non-linearity applied to each element.

# ### 3.4 (2 points)
# Describe the CNN architecture used along with training details (a flow that explains the entire training process with details on the batch_size, optimizer, loss function, model weights, learning rate, etc). Also try to infer why were these paramters and hyperparamters chosen for this specific task.
# 

# **Answer**:
# 
# 1. **Number of layers**: 121 convolution layers (DenseNet)
# 
# 2. **Output**: final fully connected layer is replaced by a layer that outputs a single value. 
# 
# 3. **Non-linearity**: sigmoid non-linearity applied at the end (after the layer in 2). 
# 
# 4. **Weights**: initialized using a model that was trained on ImageNet. 
# 
# 5. **Optimzer**: Adam ($\beta_1$ = 0.9 and $\beta_2$ = 0.999)
# 
# 6. **Batch size**: 16
# 
# 7. **Learning rate**: 0.001 (decayed by a factor of 10 when the loss plateaus after an epoch).
# 
# 7. **Loss function**: Binary cross entropy loss
# 
# **Training flow**:
# 
# - Images from pneumonia X-rays labeled as + (positive); all other images labeled - (negative). 
# 
# - Dataset divided into training (28744 patients, 98637images), validation (1672 patients, 6351 images), and test (389 patients, 420 images); no patient overalp. 
# 
# - Images downscaled to 224 $\times$ 224 and normalized before being input. 
# 
# **Why were these (hyper)parameters chosen**: Some of them (like Adam's hyperparameters) are standard/default values that most people would naturally use. The other values are likely a result of hyperparameter tuning and were obtained after trying a bunch of different things. 

# ### 3.5 (2.5 points)
# 
# How was the model evaluated? What were the metrics utilized? List down reasons of using these metrics over all others.
# 

# **Answer**:
# 
# **Evaluation against radiologists**: To compare the model against radiologists, they used the average F1 score of radiologists and the bootstrapped F1 scores of the ChexNet. They also constructed 95% confidence intervals (95CI) using bootstrapping. Since the 95CI of the difference in the model and radiologists's F1 did not include zero, the authors concluded that the model's performance was stastistically significantly better than radiologists. 
# 
# **Evaluation against previous state-of-the-art models**: To evaluate the performance of ChexNet against older models, they measured per-class AUROC. AUROC is used instead of accuracy because it measures both the true positives and the false positives. Whereas accuracy doesn't prove to be the best metric, especially when there's class imbalance. For instance, if 70% of the samples in the dataset are of class 1 and the remaining are class 0, the model could just predict everything to be class 1 and it will be right at least 70% of the time. Therefore, it's important to get the AUROC because it gives a measure to quantify the model's performance for each class individually. 

# ### 3.6 (2.5 points)
# 
# Explain model interpretation through class activation mapping. Discuss the role of Class Activation Maps (CAMs) in CheXNet.?

# **Answer**: To try and understand what areas of the X-ray were the most indicative of pneumonia, the authors generated class activation maps and overalyed them on the image after upscaling them. They did this by generating a final map that's basically the weighted sum of feature maps from the last convolution layer and their associated weights. Then they upscale this and overlay it onto the original image to see what areas of the X-ray had the largest weights associated with them to try and identify the most informative areas for penumonia prediction. 
# 
# I think it's a really cool - and conceptually simple - approach to see what the model *sees* when it is classifying the image as either having pneumonia or not. And this can also help radiologists in identifying previously missed marks/shadows on the X-ray and enhance their training in diganosing pneumonia more efficiently. 

# ### 3.7 (2 points)
# What was the kind of preprocessing the dataset went through? Explain reasons for each data transformation/preprocessing step.

# **Answer**:
# 
# - **Labeling**: The dataset was labeled with + (positive) for having pneumonia and - (negative) for not having pneumonia. The labels are important becasue this is a supervised learning task. 
# 
# - **Image re-sizing**: The images were reshaped to 224 $\times$ 224. There was some random horizontal flippling too. Reshaping is important because different images may be of different sizes, and to process them without changing their size would require different layers for each size of image. However, if we reshape them, we can use the same layers for all images. 
# 
# - **Normalization**: The images were also normalized. Normalizing helps in reducing batch effects. 
# 
# - **Horizontal flipping**: They also performed some random horizontal flipping on some of the images. This can potentially help induce some noise/randomness in the data which can help in training the model by helping it learn the different ways in which images could appear. 

# ### 3.8 (2.5 points)
# 
# In the paper CAMs (class activation mappings) are used for visualisation. Can this method be used for any CNN? Describe the architectural requirements for getting CAM visualisations.

# **Answer**: Yes, I think they can be used for any CNN. Conceptually, all they have done is take the feature map from the last image, multiply it by it's weights and get the resulting final map. This is what they call the class activation map; in this map, the regions where the weights are higher will appear to have larger values. So essentially, it's a weighted sum that places greater emphasis (during visualization) on areas that are more important for classifying pneumonia (or any other learning problem).
# 
# **Architectural requirement** To add a class activation map to any CNN, we just have to extract the filters and the feature maps generated by the final convlution layer and then multiply them together to get a map that's basically the weighted sum of the final feature map. We can then upscale it and overlay this onto the input image to generate an overlayed heatmap. 

# ## Question 4 Deep CNN design for disease classification (Total 36 points)

# In this part of the howework, we will focus on classifiying the lung disease using chest x-ray dataset provided by NIH (https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). You should be familiar with the dataset after answering question 3.
# 
# You need to use HPC for training part of this question, as your computer's CPU will not be fast enough to compute learning iterations. Please read the HPC instruction first. In case you use HPC, please have your code/scripts uploaded under the questions and provide the required plots and tables there as well. If you run the HW2 jupter script with Squash File System and Singularity on GCP, you can find the data under /images folder. We are interested in classifying pneumothorax, cardiomegaly and infiltration cases. By saying so we have 3 classes that we want to identify by modelling a deep CNN.
# 
# First, you need to work on Data_Entry_2017_v2020.csv file to identify cases/images that has infiltration, pneumothorax, and cardiomegaly. This file can be downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC

# ### 4.1 Train, Test, and Validation Sets (0.5 point)
# Write a script to read data from Data_Entry_2017.csv and process to obtain 3 sets (train, validation and test). By using 'Finding Labels' column, define a class that each image belongs to, in total you can define 3 classes:
# - 0 cardiomegaly
# - 1 pneumothorax
# - 2 infiltration
# 
# Generate a train, validation and test set by splitting the whole dataset containing specific classes (0, 1, and 2)  by 70%, 10% and 20%, respectively. Test set will not be used during modelling but it will be used to test your model's accuracy. Make sure you have similar percentages of different cases in each subset. Provide statistics of the number of classess in your subsets (you do not need to think about splitting the sets based on subjects for this homework; in general, we do not want images from the same subject to appear in both train and test sets). 
# 
# Write a .csv files defining the samples in your train, validation and test set with names: train.csv, validation.csv, and test.csv. Submit these files with your homework. 

# In[137]:


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data_Entry_2017_v2020.csv")


def assign_class(label):
    if "Cardiomegaly" in label:
        return 0
    elif "Pneumothorax" in label:
        return 1
    elif "Infiltration" in label:
        return 2
    else:
        return -1  # Exclude other cases

df["Class"] = df["Finding Labels"].apply(assign_class)
df = df[df["Class"] != -1]

train, temp = train_test_split(df, test_size=0.3, stratify=df["Class"], random_state=42)
val, test = train_test_split(temp, test_size=2/3, stratify=temp["Class"], random_state=42)

train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

#Normalizing helps in getting percentages. 
print("Train set distribution:\n", train["Class"].value_counts(normalize=True))
print("Validation set distribution:\n", val["Class"].value_counts(normalize=True))
print("Test set distribution:\n", test["Class"].value_counts(normalize=True))


# ### 4.2 Data preparation before training (2 points)
# From here on, you will use HW2_trainSet.csv, HW2_testSet.csv and HW2_validationSet.csv provided under github repo for defining train, test and validation set samples instead of the csv files you generate on question 4.1.
# 
# 
# There are multiple ways of using images as an input during training or validation. Here, you will use torch Dataset class  (http://pytorch.org/tutorials/beginner/data_loading_tutorial.html). We provided an incomplete dataloader code below. Please add your code and complete it.

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
import torch
from skimage import color

class ChestXrayDataset(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        
        image = io.imread(img_name)
    
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
    
        # If the image is grayscale, convert to 3-channel by repeating along the last axis
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        
        # Normalize the image
        image = image.astype('float32') / 255.0
    
        # Ensure the image has a single channel (batch, channels, depth, height, width)
        image = torch.tensor(image).permute(2, 0, 1)  # Adding channel dimension (1 channel)
        
        label = int(self.data_frame.iloc[idx, -1])  # Get the label
    
        # Ensure label is a tensor
        label = torch.tensor(label)
    
        # Return the dictionary
        sample = {'x': image, 'y': label}
    
        if self.transform:
            sample = self.transform(sample)
    
        return sample


# ### 4.3 CNN model definition (5 points)
# Since now we can import images for model training, next step is to define a CNN model that you will use to train disease classification task. Any model requires us to select model parameters like how many layers, what is the kernel size, how many feature maps and so on. The number of possible models is infinite, but we need to make some design choices to start.  Lets design a CNN model with 4 convolutional layers, 4 residual units (similar question 1.5) and a fully connected (FC) layer followed by a classification layer. Lets use 
# 
# -  5x5 convolution kernels (stride 1 in resnet units and stride 2 in convolutional layers)
# -  ReLU for an activation function
# -  max pooling with kernel 2x2 and stride 2 only after the convolutional layers.
# 
# Define the number of feature maps in hidden layers as: 8, 16, 32, 64, 64, 64, 128 (1st layer, ..., 7th layer). 
# 
# <img src="https://github.com/nyumc-dl/BMSC-GA-4493-Spring2025/blob/main/hws/hw2/medicalnet.png?raw=1" height="300">
# 
# Write a class which specifies this network details. 

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Disease_Classification_CNN(nn.Module):
    def __init__(self, conv_kernel: int, res_kernel: int):

        super(Disease_Classification_CNN, self).__init__()
        
        #Main convolution layers
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 8, 
                               kernel_size = conv_kernel, 
                               stride=2, 
                               padding=2)
        
        self.conv2 = nn.Conv2d(in_channels = 8, 
                               out_channels = 16, 
                               kernel_size = conv_kernel, 
                               stride=2, 
                               padding=2)
        
        self.conv3 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = conv_kernel, 
                               stride=2, 
                               padding=2)
        
        self.conv4 = nn.Conv2d(in_channels = 32, 
                               out_channels = 64, 
                               kernel_size = conv_kernel, 
                               stride=2, 
                               padding=2)

        
        #--------------------------------------------------------------
        
        #Residual unit convolution layers
        self.res_conv_1 = nn.Conv2d(in_channels = 8, 
                                    out_channels = 8, 
                                    kernel_size = res_kernel, 
                                    stride=1, 
                                    padding=2)

        self.res_conv_2 = nn.Conv2d(in_channels = 16, 
                                    out_channels = 16, 
                                    kernel_size = res_kernel, 
                                    stride=1, 
                                    padding=2)

        self.res_conv_3 = nn.Conv2d(in_channels = 32, 
                                    out_channels = 32, 
                                    kernel_size = res_kernel, 
                                    stride=1, 
                                    padding=2)
        
        self.res_conv_4 = nn.Conv2d(in_channels = 64, 
                                    out_channels = 64, 
                                    kernel_size = res_kernel, 
                                    stride=1, 
                                    padding=2)

        #--------------------------------------------------------------

        #Non-linearity
        self.relu = nn.ReLU()

        #Downsampling layer
        self.downsapling = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        #--------------------------------------------------------------

        #Can you please explain the math behind it? Like how do I get to this point for the number of input features for the fully connected layer?

        # Fully connected layer 
        self.fc1 = nn.Linear(in_features= 1024, #Because each downsampling layer is halving the dimensions. So the first layer halves 16 and 256. The second halves 8 and 128.
                             out_features = 128)  
        
        self.classification = nn.Linear(in_features = 128, 
                                        out_features = 3)  # 3 classes output

    def forward(self, x):
        
        #First convolutional layer
        #--------------------------------------------------------------
        x = self.conv1(x)
        x = self.relu(x)
        x = self.downsapling(x)

        #First residual unit
        #--------------------------------------------------------------

        identity = x
        x = self.res_conv_1(x)
        x = self.relu(x)
        x = self.res_conv_1(x)
        x = x + identity
        x = self.relu(x)

        #Second convolutional layer
        #--------------------------------------------------------------
        x = self.conv2(x)
        x = self.relu(x)
        x = self.downsapling(x)

        #Second residual unit
        #--------------------------------------------------------------
        identity = x
        x = self.res_conv_2(x)
        x = self.relu(x)
        x = self.res_conv_2(x)
        x = x + identity
        x = self.relu(x)

        #Third convolutional layer
        #--------------------------------------------------------------
        x = self.conv3(x)
        x = self.relu(x)
        x = self.downsapling(x)

        #Third residual unit
        #--------------------------------------------------------------
        identity = x
        x = self.res_conv_3(x)
        x = self.relu(x)
        x = self.res_conv_3(x)
        x = x + identity
        x = self.relu(x)

        #Fourth convolutional layer
        #--------------------------------------------------------------
        x = self.conv4(x)
        x = self.relu(x)
        x = self.downsapling(x)

        #Fourth residual unit
        #--------------------------------------------------------------
        identity = x
        x = self.res_conv_4(x)
        x = self.relu(x)
        x = self.res_conv_4(x)
        x = x + identity
        x = self.relu(x)

        #Fully connected layer
        #--------------------------------------------------------------
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.classification(x)
        log_probs = F.log_softmax(x, dim=1)
        return (torch.exp(log_probs))

Three_Disease_model = Disease_Classification_CNN(conv_kernel = 5,
                                                 res_kernel = 5)

Three_Disease_model


# ### 4.4 (2 point)
# How many learnable parameters of this model has? How many learnable parameters we would have if we replace the fully connected layer with global average pooling layer (Take a look at Section 3.2 of https://arxiv.org/pdf/1312.4400.pdf)?  

# In[3]:


for name, param in Three_Disease_model.named_parameters():
    print(f"\nLayer: {name}, Parameters: {param.numel()}")


# ### 4.5 Loss function and optimizer (2 points)
# Define an appropriate loss criterion and an optimizer using pytorch. What type of loss function is applicable to our classification problem? Explain your choice of a loss function.  For an optimizer lets use Adam for now with default hyper-parmeters.

# **Answer**: I think the cross entropy loss would be a good choice for a loss function because it's routinely used for classification problems. 

# In[4]:


from torch import optim

optimizer = optim.Adam(Three_Disease_model.parameters(), 
                          lr=0.01) #Learning rate, 
    
loss_criterion = torch.nn.CrossEntropyLoss()


# **Some background:** In network architecture design, we want to have an architecture that has enough capacity to learn. We can achieve this by using large number of feature maps and/or many more connections and activation nodes. However, having a large number of learnable parameters can easily result in overfitting. To mitigate overfitting, we can keep the number of learnable parameters of the network small either using shallow networks or few feature maps. This approach results in underfitting that model can neither model the training data nor generalize to new data. Ideally, we want to select a model at the sweet spot between underfitting and overfitting. It is hard to find the exact sweet spot. 
# 
# We first need to make sure we have enough capacity to learn, without a capacity we will underfit. Here, you will need to check if designed model in 4.3 can learn or not. Since we do not need to check the generalization capacity (overfitting is OK for now since it shows learning is possible), it is a great strategy to use a subset of training samples. Also, using a subset of samples is helpful for debugging!!!

# ### Custom Functions Block
# Every function used downstream form this block will is present here to make things neater.

# In[63]:


import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAUROC, MulticlassROC
from sklearn.metrics import confusion_matrix
import itertools

def model_training(model, train_loader, criterion, learning_rate, epochs, device = None):

    losses = []

    optimizer = optim.Adam(model.parameters(), 
                          lr=learning_rate)
    
    # Return average loss for the epoch
    for i in range(epochs):

        loss_per_batch = 0
        
        for sample in train_loader:

            if device != None:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
    
            if device == None:
                x = sample['x']
                y = sample['y']

            
            y_pred = model(x) #Forward pass

            loss = criterion(y_pred, y) #Read more about squeeze

            loss.backward() #Backward pass

            optimizer.step() #Update the weights

            optimizer.zero_grad() #Reseting the weights to 0

            loss_per_batch = loss_per_batch + loss.item()

        losses.append(loss_per_batch/len(train_loader))

        if not i % (epochs/10):
            print(f"Average training loss for epoch {i}: {losses[i]}")
    

    return losses        

#-------------------------------------------------

def model_testing(model, test_loader, criterion, device = None):

    losses = []
    correct = []
    index = 0
    predictions = []
    true_y = []
    probabilities = []  #We will use this for ROC curve
    average_loss_printed = 0

    
    model.eval()  # Set the model to evaluation mode -- Read more about this

    with torch.no_grad():

        for sample in test_loader:

            if device != None:
                x = sample['x'].to(device)
                y = sample['y'].to(device)
    
            if device == None:
                x = sample['x']
                y = sample['y']
                
            y_pred = model(x) #Forward pass

            y_pred_probs = torch.softmax(y_pred, dim=1)

            probabilities.append(y_pred_probs)

            #print(f"y_pred shape: {y_pred.shape}, y shape: {y.shape}")

            loss = criterion(y_pred, y) #Read more about squeeze

            losses.append([loss.item(), index])

            preds = torch.argmax(y_pred, dim=1)

            predictions.append(preds)

            true_y.append(y)
            
            correct_preds = (preds == y).sum().item()

            correct.append([correct_preds, index])
            
            index = index + 1

            #print("Batch:", index, "Loss:", loss.item(), "Correct:", correct_preds, "Wrong:", len(y)-(correct_preds))

            average_loss_printed += loss.item()

    #print("Average loss for the test set:", average_loss_printed/len(test_loader))

    probabilities = torch.cat(probabilities)

    predictions = torch.cat(predictions)

    true_y = torch.cat(true_y)

    return predictions, true_y, probabilities, (average_loss_printed/len(test_loader))

#------------------------------------

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

#-----------------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

#def cm(y_true, y_pred):
#    cnf_matrix = confusion_matrix(y_true, y_pred)
#    np.set_printoptions(precision=2)
#    plt.figure()
#    class_names = ['0: Cardiomegaly','1: Pneumothorax','2: Infiltration']
#    plot_confusion_matrix(cnf_matrix, classes=class_names,
#                          title='Confusion matrix')

from sklearn.metrics import confusion_matrix

def cm(y_true, y_pred):
    # Specify the label order explicitly.
    label_order = [0, 1, 2]
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    np.set_printoptions(precision=2)
    plt.figure()
    class_names = ['0: Cardiomegaly','1: Pneumothorax','2: Infiltration']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix')


# ### 4.6 Train the network on a subset (5 points)
# Lets use a script to take random samples from train set (HW2_trainSet.csv), lets name this set as HW2_randomTrainSet. Choose random samples from validation set (HW2_validationSet.csv), lets name this set as HW2_randomValidationSet. You used downsampling of images from 1024x1024 size to 64x64 in the Lab 4. This was fine for learning purpose but it will significantly reduce the infomation content of the images which is important especially in medicine. In this Homework, you MUST use original images of size 1024x1024 as the network input. 

# In[150]:


# get samples from HW2_trainSet.csv
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('HW2_trainSet.csv')
_ , X_random, _, _ = train_test_split(df, df.Class, test_size=0.1, random_state=0)
print('Selected subset class frequencies\n',X_random['Class'].value_counts())
X_random.to_csv('HW2_randomTrainSet.csv',index=False)

df = pd.read_csv('HW2_validationSet.csv')
_ , X_random, _, _ = train_test_split(df, df.Class, test_size=0.1, random_state=0)
print('Selected subset class frequencies\n',X_random['Class'].value_counts())
X_random.to_csv('HW2_randomValidationSet.csv',index=False)


# Use the random samples generated and write a script to train your network. Using the script train your network using your choice of weight initialization strategy. In case you need to define other hyperparameters choose them empirically, for example batch size. Plot average loss on your random sample set per epoch. (Stop the training after at most ~50 epochs).

# In[19]:


# Define paths
train_csv = "HW2_randomTrainSet.csv"
val_csv = "HW2_randomValidationSet.csv"
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"

# Create dataset objects
train_dataset_subset = ChestXrayDataset(train_csv, root_dir)
val_dataset_subset = ChestXrayDataset(val_csv, root_dir)

# Create DataLoaders
train_loader_subset = DataLoader(train_dataset_subset, batch_size=32, shuffle=True, num_workers=14)
val_loader_subet = DataLoader(val_dataset_subset, batch_size=32, shuffle=False, num_workers=14)


# In[181]:


epochs_for_nn = 10
learning_rate_for_nn = 0.01
gpu = torch.device('cuda:0')

reset_weights(Three_Disease_model)

average_losses = model_training(model = Three_Disease_model.to(gpu), 
                                train_loader = train_loader_subset, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = learning_rate_for_nn, 
                                epochs = epochs_for_nn,
                                device = gpu)

import matplotlib.pyplot as plt

# Create figure and axis for better visualization
plt.plot(range(epochs_for_nn), 
         average_losses, 
         color='darkmagenta')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Average Loss for each Epoch")

# Display the plot
plt.show()


# In[184]:


subset_predictions = model_testing(model = Three_Disease_model.to(gpu), 
                                   test_loader = val_loader_subet, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

print(f"The average loss for the test set is: {subset_predictions[3]}")


# ### 4.7 Analysis of training using a CNN model (2 points)
# Describe your findings. Can your network learn from small subset of random samples? Does CNN model have enough capacity to learn with your choice of emprical hyperparameters?
# -  If yes, how will average loss plot will change if you multiply the learning rate by 15?
# -  If no, how can you increase the model capacity? Increase your model capacity and train again until you find a model with enough capacity. If the capacity increase is not sufficient to learn, think about empirical parameters you choose in designing your network and make some changes on your selection. Describe what type of changes you made to your original network and how can you manage this model to learn.

# **Answer**: The model does not seem to have enough capacity to learn; there's huge variations in the loss reported for each epoch. Some of the ways that we could try to improve the learning capacity was using a weighted loss function (I tried it but it didn't work - data not shown), or give it a larger subset of the data. 
# 
# - Let's try increasing the learning rate by 15 times to see how it changes the average loss plot:

# In[185]:


epochs_for_nn = 10
learning_rate_for_nn = 0.01
gpu = torch.device('cuda:0')

reset_weights(Three_Disease_model)

average_losses = model_training(model = Three_Disease_model.to(gpu), 
                                train_loader = train_loader_subset, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = learning_rate_for_nn*15, 
                                epochs = epochs_for_nn,
                                device = gpu)

import matplotlib.pyplot as plt

# Create figure and axis for better visualization
plt.plot(range(epochs_for_nn), 
         average_losses, 
         color='darkmagenta')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Average Loss for each Epoch with 15 x Learning Rate")

# Display the plot
plt.show()


# In[187]:


subset_predictions_15_lr = model_testing(model = Three_Disease_model.to(gpu), 
                                   test_loader = val_loader_subet, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

print(f"The average loss for the test set is: {subset_predictions_15_lr[3]}")


# The model seems to be learning just as bad as before with 15 $\times$ the learning rate.
# 
# **Additionally:** The random subset of data only has 2 classes, however, the model is trained to output three classes. That could explain why it's not able to learn very well, because while it expects outputs of three classes, its loss function might be penalizing it over and over for predicting the third class when  in fact it should only output 2 classes. Therefore, if we change the output to 2 classes perhaps the model would learn better. 

# ### 4.8 Hyperparameters (2.5 points)
# Now, we will revisit our selection of CNN model architecture, training parameters and so on: i.e. hyperparameters. In your investigations, define how you will change the hyperparameter in the light of model performance using previous hyperparameters. Provide your rationale choosing the next hyperparameter. Provide learning loss and accuracy curves, and model performance in HW2_randomValidationSet. You will use macro AUC as the performance metric for comparing CNN models for disease classification task.  Report macro AUC for each CNN model with different hyperparameters (Check http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings).
# 
# Investigate the effect of learning rate and batch size in the model performance (try atleast 5 learning rates and 3 batch sizes) and select optimal values for both. You only need to put your best result here.

# In[197]:


epochs_for_nn = 5
gpu = torch.device('cuda:0')
num_classes = 3
roc_metric = MulticlassROC(num_classes=num_classes, average=None)
auroc_metric = MulticlassAUROC(num_classes=num_classes, average=None)

#------------------------------------------------------------------

learning_rate_array = np.array([0.001, 0.01, 0.15, 1.5, 15])

for i in range(len(learning_rate_array)):

    reset_weights(Three_Disease_model)

    print(f"\nTraining model at the learning rate: {learning_rate_array[i]} with the batch size of 32")

    average_losses = model_training(model = Three_Disease_model.to(gpu), 
                                train_loader = train_loader_subset, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = learning_rate_array[i], 
                                epochs = epochs_for_nn,
                                device = gpu)

    print(f"Testing model trained at the learning rate: {learning_rate_array[i]}")

    lr_tuning_predictions = model_testing(model = Three_Disease_model.to(gpu), 
                                   test_loader = val_loader_subet, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

    auroc_lr = auroc_metric(lr_tuning_predictions[2].to("cpu"), lr_tuning_predictions[1].to("cpu"))


    print(f"The average loss for the test set at the learning rate {learning_rate_array[i]} is: {lr_tuning_predictions[3]}")
    print(f"The AUROC for the test set at the learning rate {learning_rate_array[i]} is: {np.mean(np.array(auroc_lr))}")
    print("---------------------------------------------------------")

#------------------------------------------------------------------

batch_size_array = np.array([16, 32, 64, 128, 256]) #Data loader only accepts integers. 

for i in range(len(batch_size_array)):

    tuning_train_loader_subset = DataLoader(train_dataset_subset, batch_size=int(batch_size_array[i]), shuffle=True, num_workers=14)
    tuning_val_loader_subet = DataLoader(val_dataset_subset, batch_size=int(batch_size_array[i]), shuffle=False, num_workers=14)

    reset_weights(Three_Disease_model)

    print(f"\nTraining model at the batch size of: {batch_size_array[i]} with a learning rate of 0.01")

    average_losses = model_training(model = Three_Disease_model.to(gpu), 
                                train_loader = tuning_train_loader_subset, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = 0.01, 
                                epochs = epochs_for_nn,
                                device = gpu)

    print(f"Testing model trained at the batch size of: {batch_size_array[i]}")

    batch_tuning_predictions = model_testing(model = Three_Disease_model.to(gpu), 
                                   test_loader = tuning_val_loader_subet, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

    auroc_batch = auroc_metric(batch_tuning_predictions[2].to("cpu"), batch_tuning_predictions[1].to("cpu"))
    
    print(f"The average loss for the test set at the learning rate {batch_size_array[i]} is: {batch_tuning_predictions[3]}")
    print(f"The AUROC for the test set at the batch size {batch_size_array[i]} is: {np.mean(np.array(auroc_batch))}")
    print("---------------------------------------------------------")


# ### 4.9 Train the network on the whole dataset (4 points)
# After question 4.7, you should have a network which has enough capacity to learn and you were able to debug your training code so that it is now ready to be trained on the whole dataset. Use the best batch size and learning rate from 4.8. Train your network on the whole train set (HW2_trainSet_new.csv) and check the validation loss on the whole validation set (HW2_validationSet_new.csv) in each epoch. Plot average loss and accuracy on train and validation sets. Describe your findings. Do you see overfitting or underfitting to train set? What else you can do to mitigate it?

# In[97]:


def assign_class(label):
    if "Cardiomegaly" in label:
        return 0
    elif "Pneumothorax" in label:
        return 1
    elif "Infiltration" in label:
        return 2
    else:
        return 3 

train_csv = "HW2_trainSet_new.csv"
val_csv = "HW2_validationSet_new.csv"
test_csv = "HW2_testSet_new.csv"
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"

train_df = pd.read_csv(train_csv)
train_df["Class"] = train_df["Finding Labels"].apply(assign_class)

val_df = pd.read_csv(val_csv)
val_df["Class"] = val_df["Finding Labels"].apply(assign_class)

test_df = pd.read_csv(test_csv)
test_df["Class"] = test_df["Finding Labels"].apply(assign_class)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)


# In[29]:


train_csv = "HW2_trainSet_new.csv"
val_csv = "HW2_validationSet_new.csv"
test_csv = "HW2_testSet_new.csv"
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"

# Create dataset objects
full_train_dataset = ChestXrayDataset(train_csv, root_dir)
full_val_dataset = ChestXrayDataset(val_csv, root_dir)

# Create DataLoaders
full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, num_workers=14)
full_val_loader = DataLoader(full_val_dataset, batch_size=64, shuffle=False, num_workers=14)


# In[30]:


print('Training set class frequencies\n',pd.read_csv(train_csv)['Class'].value_counts())
print('Validation set class frequencies\n',pd.read_csv(val_csv)['Class'].value_counts())
print('Test set class frequencies\n',pd.read_csv(test_csv)['Class'].value_counts())


# In[15]:


gpu = torch.device('cuda:0')

class_counts = torch.tensor([pd.read_csv(train_csv)['Class'].value_counts()[0],
                           pd.read_csv(train_csv)['Class'].value_counts()[1],
                           pd.read_csv(train_csv)['Class'].value_counts()[2]])

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize weights
print("Class Weights:", class_weights)

weighted_cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(gpu))


# In[31]:


gpu = torch.device('cuda:0')
epochs_for_nn = 10
learning_rate_for_nn = 0.01

reset_weights(Three_Disease_model)

average_losses = model_training(model = Three_Disease_model.to(gpu), 
                                train_loader = full_train_loader, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = learning_rate_for_nn, 
                                epochs = epochs_for_nn,
                                device = gpu)

import matplotlib.pyplot as plt

# Create figure and axis for better visualization
plt.plot(range(epochs_for_nn), 
         average_losses, 
         color='darkmagenta')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Average Loss for each Epoch")

# Display the plot
plt.show()


# In[18]:


class_counts_val = torch.tensor([pd.read_csv(val_csv)['Class'].value_counts()[0],
                           pd.read_csv(val_csv)['Class'].value_counts()[1],
                           pd.read_csv(val_csv)['Class'].value_counts()[2]])

class_weights_val = 1.0 / class_counts_val
class_weights_val = class_weights_val / class_weights_val.sum()  # Normalize weights
print("Class Weights:", class_weights_val)

weighted_cross_entropy_loss_val = torch.nn.CrossEntropyLoss(weight=class_weights_val.to(gpu))


# In[32]:


full_val_predictions = model_testing(model = Three_Disease_model.to(gpu), 
                                         test_loader = full_val_loader, 
                                         criterion = torch.nn.CrossEntropyLoss(),
                                     device = gpu)

print(f"The average loss for the validation set is: {full_val_predictions[3]}")


# Since the training and validation losses are pretty close, it would suggest that the model is fitting the data, and not overfitting it. However, we can regularize the model to further reduce the loss. 

# ### 4.10 Experiments with Resnet18
# 
# Let's use Resnet18 on our dataset and see how it performs. We can import the standard architectures directly using PyTorch's torchvison.models module. Refer to https://pytorch.org/docs/stable/torchvision/models.html to see all available models in PyTorch. You'll later, in this course, learn about a convenient and useful concept known as Transfer Learning. For now, we will  use the Resnet18 and train the architecture from scratch without any pre-training. Here is the link for the ResNet paper: https://arxiv.org/pdf/1512.03385.pdf .

# #### 4.10.a (2 Point)
# 
# What is the reason of using 1x1 convolutions before 3x3 convolutions in the resnet architecture?

# **Answer**: This makes the 3 $\times$ 3 layer the bottleneck. The 1 $\times$ 1 layer reduces the dimensionality. And since there's only 1 parameter in the kernel, the process is fast. The 3 $\times$ 3 layer then has a lower dimension map to work on because the 1 $\times$ 1 layer extracts all the important information while reducing the dimensions. This way, the computations take less time and improve the efficiency of the model. 

# #### 4.10.b Train the ResNet18 on the whole dataset

# We provide a new dataset class and a few additional transformations to the data for this new architecture. We have a new dataset class as ResNet18 architectures expect 3 channels in their primary input and other reasons which you'll later come to know - after the lecture on transfer learning. Nevertheless, for our case, we use them to reduce the required GPU usage as the Resnet18 architecture is significantly complex and GPU memory-intensive architecture than the CNN implemented above.

# In[67]:


from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# torchvision models are trained on input images normalized to [0 1] range .ToPILImage() function achives this
# additional normalization is required see: http://pytorch.org/docs/master/torchvision/models.html

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ChestXrayDataset_ResNet(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = load_data_and_get_class(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        
        image = io.imread(img_name)
        
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
    
    # If the image is grayscale, convert to 3-channel by repeating along the last axis
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        

        image_class = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)
            
        sample = {'x': image, 'y': image_class}

        return sample

def load_data_and_get_class(path_to_data):
    data = pd.read_csv(path_to_data)
    # Define the explicit mapping. Adjust the keys if needed.
    class_mapping = {"Cardiomegaly": 0, "Pneumothorax": 1, "Infiltration": 2}
    data['Class'] = data['Finding Labels'].map(class_mapping)
    return data


# #### 4.10.c Architecture modification (4.5 points) 
# In this question you need to develop a CNN model based on Resnet18 architecture. Please import the original ResNet18 model from PyTorch models (You can also implement this model by your own using the resnet paper). Modify the architecture so that the model will work with full size 1024x1024 image inputs and 3 classes of our interest:
# - 0 cardiomegaly
# - 1 pneumothorax
# - 2 infiltration
# 
# Make sure the model you developed uses random weights!

# In[34]:


from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model
resnet_18 = models.resnet18(weights=None)  # No pre-training
num_ftrs = resnet_18.fc.in_features  # Get number of input features for FC layer

# Modify last fully connected layer for our dataset
num_classes = 3
resnet_18.fc = nn.Linear(num_ftrs, num_classes)

resnet_18 = resnet_18.to(device)


# #### 4.10.d Train the network on the whole dataset (4.5 points)
# Similar to question 4.7 train the model you developed in question 4.10.b on the whole train set (HW2_trainSet_new.csv) and check the validation loss on the whole validation set (HW2_validationSet_new.csv) in each epoch. Plot average loss and accuracy on train and validation sets. Describe your findings. Do you see overfitting or underfitting to train set? What else you can do to mitigate it?

# In[35]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np

train_csv = "HW2_trainSet_new.csv"
val_csv = "HW2_validationSet_new.csv"
test_csv = "HW2_testSet_new.csv"
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"

# Load datasets
resnet_train_dataset = ChestXrayDataset_ResNet(csv_file=train_csv, root_dir=root_dir, transform=train_transform)
resnet_val_dataset = ChestXrayDataset_ResNet(csv_file=val_csv, root_dir=root_dir, transform=validation_transform)

# Create data loaders
batch_size = 64
resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=batch_size, shuffle=False, num_workers=14)


# In[36]:


gpu = torch.device('cuda:0')

epochs_for_nn = 10
learning_rate_for_nn = 0.01

average_losses = model_training(model = resnet_18, 
                                train_loader = resnet_train_loader, 
                                criterion = torch.nn.CrossEntropyLoss(), 
                                learning_rate = learning_rate_for_nn, 
                                epochs = epochs_for_nn,
                                device = gpu)

import matplotlib.pyplot as plt

# Create figure and axis for better visualization
plt.plot(range(epochs_for_nn), 
         average_losses, 
         color='darkmagenta')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Average Loss for each Epoch")

# Display the plot
plt.show()


# In[45]:


resnet_predictions = model_testing(model = resnet_18, 
                                   test_loader = resnet_val_loader, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

print(f"The average loss for the validation set is: {resnet_predictions[3]}")


# The ResNet18 model seems to be fitting the data better than the CNN that we created above. Since the loss for the training and the validation set is pretty similar, it suggests that the model is not overfitting. 
# 
# However, we can try regularization to improve the model's performance. We can also use a weighted loss function because there's a large class imbalance (I tried it but it didn't work - data not shown). 
# 
# Additionally, we can try modifying the architecture a little to add additional layers, non-linearity, and perhaps even auxilliary classifiers to improve the model's prediction.

# ## Question 5 Analysis of the results from two networks trained on the full dataset (Total 5 points)
# Use the validation loss to choose models from question 4.9 (model1) and question 4.10 (model2) (these models are trained on the full dataset and they learned from train data and generalized well to the validation set). 

# ### 5.1 Model selection by performance on test set (5 Points)
# Using these models, plot confusion matrix and ROC curve for the disease classifier on the test set (HW2_TestSet_new.csv). Report AUC for this CNN model as the performance metric. You will have two confusion matrices and two ROC curves to compare model1 and model2.

# In[46]:


root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"
test_csv = "HW2_testSet_new.csv"

#----
my_model_test_dataset = ChestXrayDataset(test_csv, root_dir)
my_model_test_loader = DataLoader(my_model_test_dataset, batch_size=64, shuffle=False, num_workers=14)

#------

# Load datasets
ResNet_test_dataset = ChestXrayDataset_ResNet(csv_file=test_csv, root_dir=root_dir, transform=train_transform)
ResNet_test_loader = DataLoader(ResNet_test_dataset, batch_size=64, shuffle=False, num_workers=14)


# In[47]:


class_counts_test = torch.tensor([pd.read_csv(test_csv)['Class'].value_counts()[0],
                           pd.read_csv(test_csv)['Class'].value_counts()[1],
                           pd.read_csv(test_csv)['Class'].value_counts()[2]])

class_weights_test = 1.0 / class_counts_test
class_weights_test = class_weights_test / class_weights_test.sum()  # Normalize weights
print("Class Weights:", class_weights_test)

weighted_cross_entropy_loss_test = torch.nn.CrossEntropyLoss(weight=class_weights_test.to(gpu))


# In[48]:


full_test_predictions = model_testing(model = Three_Disease_model.to(gpu), 
                                         test_loader = my_model_test_loader, 
                                         criterion = torch.nn.CrossEntropyLoss(),
                                     device = gpu)

print(f"\nThe average loss of my CNN model for the test set is: {full_test_predictions[3]}")


# In[68]:


resnet_predictions_test = model_testing(model = resnet_18, 
                                   test_loader = ResNet_test_loader, 
                                   criterion = torch.nn.CrossEntropyLoss(),
                                   device = gpu)

print(f"\nThe average loss of ResNet for the test set is: {resnet_predictions_test[3]}")


# #### ROC for the CNN I created

# In[41]:


import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = full_test_predictions[1].to("cpu")
y_probas = full_test_predictions[2].to("cpu")
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()


# #### ROC for ResNet18

# In[69]:


import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = resnet_predictions_test[1].to("cpu")
y_probas = resnet_predictions_test[2].to("cpu")
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()


# #### Confusion Matrix for ResNet18

# In[71]:


cm(resnet_predictions_test[0].to("cpu"), resnet_predictions_test[1].to("cpu"))


# #### Confusion Matrix for the CNN I created

# In[72]:


cm(full_test_predictions[0].to("cpu"), full_test_predictions[1].to("cpu"))


# ##  6 Bonus Questions (Maximum 12 points)
# 
# **Note:** this section is optional.

# ### 6.1 Understanding the network (Bonus Question maximum 5 points)
# 
# Even if you do both 6.1.a and 6.1.b, the max points for this question is 5.

# #### 6.1.a Occlusion (5 points)
# Using the best performing model (choose the model using the analysis you performed on question 5.1), we will figure out where our network gathers infomation to decide the class for the image. One way of doing this is to occlude parts of the image and run through your network. By changing the location of the ocluded region we can visualize the probability of image being in one class as a 2-dimensional heat map. Using the best performing model, provide the heat map of the following images: HW2_visualize.csv. Do the heap map and bounding box for pathologies provide similar information? Describe your findings.
# Reference: https://arxiv.org/pdf/1311.2901.pdf

# In[76]:


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Parameters
occlusion_size = 50  # size of the occlusion square
stride = 20          # sliding window stride
target_class = 1     # change this to the target class index you want to analyze

# Define image transformation (should match the one used in training)
transform = transforms.Compose([
    transforms.Resize((896, 896)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def occlusion_heatmap(model, image, occlusion_size, stride, target_class, occlusion_value=0):

    model.eval()
    _, H, W = image.shape
    heatmap = np.zeros(((H - occlusion_size) // stride + 1,
                        (W - occlusion_size) // stride + 1))
    
    original_image = image.clone()
    
    with torch.no_grad():
        for i, h in enumerate(range(0, H - occlusion_size + 1, stride)):
            for j, w in enumerate(range(0, W - occlusion_size + 1, stride)):
                occluded_image = original_image.clone()
                # Occlude a region in all channels
                occluded_image[:, h:h+occlusion_size, w:w+occlusion_size] = occlusion_value
                occluded_image = occluded_image.unsqueeze(0)  # add batch dim
                
                # Forward pass
                output = model(occluded_image)
                probs = F.softmax(output, dim=1)
                # Record the probability of the target class
                heatmap[i, j] = probs[0, target_class].item()
    
    return heatmap

# Load the CSV file containing the paths to the images
visualize_df = pd.read_csv('HW2_visualize.csv')

# Assuming the CSV contains a column 'image_path'
# and that paths are relative to a known root directory:
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"
image_paths = [root_dir + path for path in visualize_df['Image Index'].tolist()]

# Process each image and generate a heat map
for img_path in image_paths:
    # Load image using PIL
    pil_img = Image.open(img_path).convert('RGB')
    input_img = transform(pil_img).to(gpu)  # shape: (C, H, W)
    
    # Generate occlusion heatmap
    heatmap = occlusion_heatmap(resnet_18, input_img, occlusion_size, stride, target_class)
    
    # Plot the original image and heatmap
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(f"Occlusion Heatmap (Class {target_class})")
    plt.colorbar()
    
    plt.show()


# In the third picture, it looks like the model is focusing on the empty space between the arm and the side of the abdomen. This can probably explain why the model has a high loss. And for future reference, we can try to crop the images in a way that only the torso is included with no empty (dark) spaces that could throw off the model. 

# In[ ]:


# you can use the code from: https://github.com/thesemicolonguy/convisualize_nb/blob/master/cnn-visualize.ipynb 


# #### 6.1.b GradCAM (5 points)
# An alternative approach to model interpretation is gradcam. Go through https://arxiv.org/pdf/1610.02391.pdf and create heatmaps of images in HW2_visualize.csv using this method. Repeat the analysis in 6.1.a and also compare the time-taken to generate occlusions and gradcams

# In[85]:


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# Load the best-performing model
resnet_18.eval()

# Grad-CAM requires the layer name; find the last convolutional layer
target_layer = "layer4"  # Modify this based on your model architecture

# Define transformation (should match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((896, 896)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load image paths from CSV
visualize_csv = 'HW2_visualize.csv'
root_dir = "/scratch/ma8308/Deep_Learning/HW2/images/images/"
visualize_df = pd.read_csv(visualize_csv)
image_paths = [os.path.join(root_dir, path) for path in visualize_df['Image Index'].tolist()]

# Define target class
target_class = 1  # Adjust based on pathology class index

# Initialize Grad-CAM method
cam_extractor = GradCAM(resnet_18, target_layer)

# Store execution times for comparison
occlusion_times = []
gradcam_times = []

for img_path in image_paths:
    # Load and preprocess the image
    pil_img = Image.open(img_path).convert('RGB')
    input_img = transform(pil_img).unsqueeze(0).to(gpu)  # Add batch dimension
    
    ### 1️⃣ Grad-CAM Processing
    start_time = time.time()
    
    scores = resnet_18(input_img)
    predicted_class = scores.argmax(dim=1).item()  # Get predicted label
    
    # Generate Grad-CAM heatmap
    activation_map = cam_extractor(predicted_class, scores)
    end_time = time.time()
    gradcam_times.append(end_time - start_time)
    
    # Convert to PIL Image and overlay
    heatmap = activation_map[0].squeeze().to("cpu").numpy()
    result = overlay_mask(pil_img, Image.fromarray((heatmap * 255).astype(np.uint8), mode='L'), alpha=0.5)

    # Plot Grad-CAM heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(pil_img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(result)
    ax2.set_title("Grad-CAM Heatmap")
    ax2.axis('off')

    plt.show()

# Print comparison of time taken
print(f"Average time for Grad-CAM: {np.mean(gradcam_times):.2f} seconds per image")


# The Grad-CAMs were a lot faster to compute. 

# ### 6.2 Tiling and CNNs (Bonus Question 7 points)

# When using CNNs it may be helpful to first tile the image, especially for segmentation and object detection tasks. Focus on the "Invasive Ductal Carcinoma Segmentation Use Case" section of this [paper](https://www.sciencedirect.com/science/article/pii/S2153353922005478?via%3Dihub#tbl1). The data is avaliable [here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data).

# #### 6.2.a (0.5 points)
# 
# Why is it helpful to tile an image and use the tiles as input for a CNN for segmentation?

# **Answer**: For a segmentation task, the boundaries need to be clearly defined. However, with tumor images, the tumors are relatively small compared to the tissue around them. Looking at the whole slide image might confuse the model and make it learn the boundaries of vessels or sub-tissues as being the boundaries associated with carcinomas. 
# 
# On the other hand, tiling the image can allow the model to learn the tumor boundaries much more effectively because it will only have specific boundaries in an image to focus on. This way, the model will be more effective at segmentation. 

# #### 6.2.b (0.5 points)
# 
# Describe the hyperparameters that are introduced when you tile an image.

# **Answer**: We would need to think about what size the tiles should be. And we would need to tile an image in a way that doesn't overlap two tiles - so the "tiling" stride will also be another parameter. 

# #### 6.2.c (0.5 points)
# 
# What are some metrics that can be used to evaluate segmenation of the full image (when tiles are recombined)?

# **Answer**: We can potentially measure the mean distance between the predicted boundary and the predicted boundary for one tile. Or after we recombine the the whole thing. That will give us an estimate of how far off the model was from the original boundary - but I don't know if there's a loss function for this, or how to define the real boundary to measure the distance against. Perhaps we can use the coordinates of the boundary as the "real" boundary, and measure the distance between those coordinates and the predicted coordinates. That would sort of become like euclidean distance and then we can use that as the loss function. 

# #### 6.2.d (4 points)
# 
# Load the data, train a CNN, and evaluate the performance on the dataset.
# 
# **Note:** due to the size of this dataset, feel free to sample only part of the dataset to use to train and evaluate your model. Just please make sure all classes are represented, and that you do not train and test on the same patients.

# In[53]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

# Define the main data path
data_path = "/scratch/ma8308/Deep_Learning/breast_histo/"

# Collect image paths and labels from all patient folders
all_image_paths = []
all_labels = []

for patient_id in os.listdir(data_path)[:100]:
    patient_path = os.path.join(data_path, patient_id)
    if os.path.isdir(patient_path):  # Ensure it's a directory
        for class_label in [0, 1]:  # 0 = No IDC, 1 = IDC
            class_path = os.path.join(patient_path, str(class_label))
            if os.path.exists(class_path):  # Ensure sub-folder exists
                for img_name in os.listdir(class_path):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_labels.append(class_label)

# Split dataset into train and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_image_paths[:int(len(all_image_paths)/2)], all_labels[:int(len(all_image_paths)/2)], 
    test_size=0.3, stratify=all_labels[:int(len(all_image_paths)/2)], random_state=42
)

# Define Dataset class
class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Datasets
train_dataset = BreastCancerDataset(train_paths, train_labels, transform=transform)
test_dataset = BreastCancerDataset(test_paths, test_labels, transform=transform)
validation_dataset = BreastCancerDataset(all_image_paths[int(len(all_image_paths)/2):], 
                                         all_labels[int(len(all_image_paths)/2):], transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 14)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 14)

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # No pre-training
num_ftrs = model.fc.in_features
num_classes = 2  # Binary classification
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluation
model.eval()
y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(probabilities.cpu().numpy())

# Compute AUROC
def plot_auroc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# Compute and plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["No IDC", "IDC"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Plot results
plot_auroc(y_true, y_scores)
plot_confusion_matrix(y_true, y_pred)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["No IDC", "IDC"]))


# In[54]:


val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers = 14)

model.eval()
y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(probabilities.cpu().numpy())

# Plot results
plot_auroc(y_true, y_scores)
plot_confusion_matrix(y_true, y_pred)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["No IDC", "IDC"]))


# #### 6.2.e (1.5 points)
# 
# Select a patch of 7x7 images and predict their classification. Then display them all together as one image, and denote the patches that are predicted as IDC. Diplay another image that denotes that patches that are IDC.

# In[55]:


import torchvision.utils as vutils

# Select 49 random images (7x7 grid)
num_patches = 49  # 7x7 grid
indices = np.random.choice(len(test_dataset), num_patches, replace=False)

# Get the images and labels
selected_images = [test_dataset[i][0] for i in indices]
selected_labels = [test_dataset[i][1].item() for i in indices]

# Stack images into a batch for model prediction
images_batch = torch.stack(selected_images).to(device)

# Run model inference
model.eval()
with torch.no_grad():
    outputs = model(images_batch)
    _, predictions = torch.max(outputs, 1)

# Convert to NumPy for easy processing
predictions = predictions.cpu().numpy()
selected_labels = np.array(selected_labels)

# Create a grid of images
grid_img = vutils.make_grid(images_batch.cpu(), nrow=7, padding=2, normalize=True)

# Create a mask for IDC (malignant) predictions
idc_mask = predictions == 1  # Boolean mask where IDC is predicted

# Create a binary mask image for IDC regions
idc_highlight = torch.zeros_like(images_batch)  # Black image
idc_highlight[idc_mask] = 1  # Set IDC predicted areas to white

# Convert grid images to NumPy for visualization
grid_img_np = np.transpose(grid_img.numpy(), (1, 2, 0))

# Plot the image grid with predictions
plt.figure(figsize=(10, 10))
plt.imshow(grid_img_np)
plt.axis('off')
plt.title("7x7 Image Patches with IDC Highlighted")
plt.show()

# Plot IDC highlight mask separately
highlight_grid = vutils.make_grid(idc_highlight.cpu(), nrow=7, padding=2, normalize=True)
highlight_grid_np = np.transpose(highlight_grid.numpy(), (1, 2, 0))

plt.figure(figsize=(10, 10))
plt.imshow(highlight_grid_np, cmap="Reds")  # Use red to highlight IDC areas
plt.axis('off')
plt.title("IDC Patches Highlighted in Red")
plt.show()

