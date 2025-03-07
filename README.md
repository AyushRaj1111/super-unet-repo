# Super U-Net Architecture

Super U-Net was designed by modifying each U-Net core component (i.e., encoder, decoder, and skip connections) to extract detailed spatial information and integrate the concatenated feature maps from the encoder. The Super U-net encoder block is a modified version of the Residual units developed by He et al. It employs residual blocks to minimize the degradation problem occurring in deep networks. Fusion upsampling and dynamic receptive field modules were developed as part of Super U-net. Fusion upsampling leverages squeeze and excitation to aggregate divergent feature maps into similar feature representations. The fusion upsampling module was used to modify skip connections. The dynamic receptive field module allows the network to determine the best kernel size for the current segmentation task at each iteration in training. Dynamic kernel selection grants the network parallel paths of varying kernel size and allows for the extraction of multiscale spatial information. After integrating each module, essential semantic and spatial information is preserved. Super U-net is designed to have a moderate number of network parameters (4.2 million) when compared to the typical million network parameters of other U-Nets.

## Residual Block

Super U-Net uses the residual connections proposed by He et al. as the central convolution component. The first layer is a convolutional operation with a kernel size of 3 × 3 pixels. After the convolution layer, a batch normalization layer is applied and followed by the nonlinear rectified linear unit (ReLU) activation function. Next, the feature map undergoes an additional convolutional layer prior to having the initial feature map added to the image. Thereafter, the combined feature map receives batch normalization and a ReLU activation function. As the feature maps cascade down each layer of the encoder, the number of filters doubles on every successive layer (i.e., 8, 16, 32, 64, 128). After the residual operations, the feature map is then recalibrated by implementing a squeeze and excitation module to form one branch of the fusion upsampling module.

## Fusion Upsampling and Concatenation Module

The recalibrated feature map from the encoder immediately passes across the long skip connection to form the encoded branch of the upsampled fusion module. This module is used to integrate the feature maps from the encoder with the upsampled feature maps of the decoder. The encoder and decoder feature maps are fed into a squeeze and excitation module before concatenation. The encoded feature map undergoes a squeeze and excitation operation prior to the skip connection, which allows for the recalibrated feature map to assist the lower levels of the encoder network with the extraction of meaningful features. The decoded feature map is recalibrated and then upsampled to allow for a better distribution of features. The feature maps are “squeezed” with a global average pooling operation, and then the features are “excited,” allowing for an adaptive recalibration of channel-wise dependencies. Next, the features enter a multilayer perceptron (MLP), where the first layer contains more nodes than the input layers. A ReLU activation function is applied to the aggregated features before the features pass through an additional fully connected layer. Next, a sigmoid activation function is applied to the features followed by reshaping and multiplying the output channel-wise across the input. As the features flow upward in the decoder, the number of feature channels is reduced. After each reduction, fewer feature maps are retained. Adaptive recalibration allows the interaction between channels to be better represented when upsampled and concatenated with the corresponding feature map. Squeeze and excitation operations ensure that meaningful features are retained as the number of channels is reduced. After concatenation, the fused feature maps enter a residual block.

## Dynamic Receptive Field Module

The dynamic receptive field module creates three independent paths of unique kernel sizes and allows the network to determine the optimal path. Specifically, the receptive field module has three parallel routes with kernel sizes of 1 × 1, 3 × 3, and 5 × 5. After this initial convolutional layer, each path goes through another convolutional layer with kernel sizes equal to the previous layer. Each convolution path proceeds to dilated convolution layer with a dilation rate equivalent to their kernel size (i.e., Conv3 × 3 has a dilation rate of 3). The kernel size of 1 provides a baseline for the network to refine its feature maps. Each path is then concatenated before undergoing a final convolutional layer. Thereafter, the output is concatenated onto the initial feature map. By granting the network the freedom of choice in its path, the network can optimally learn multiscale contextual information.

## Instructions to Run the Code

1. Clone the repository:
   ```
   git clone https://github.com/githubnext/workspace-blank.git
   cd workspace-blank
   ```

2. Install the required dependencies:
   ```
   pip install tensorflow numpy
   ```

3. Open the Jupyter Notebook:
   ```
   jupyter notebook super_unet.ipynb
   ```

4. Run all cells in the notebook to define the Super U-Net model, compile it, and train it with dummy training data.

## Datasets Used for Training and Validation

Super U-Net was trained and evaluated using four publicly available datasets: (1) Digital Retinal Images for Vessel Extraction (DRIVE), (2) Kvasir-SEG, (3) Child Heart and Health Study in England (CHASE DB1), and (4) International Skin Imaging Collaboration (ISIC).

1. **DRIVE**: This dataset was collected from a diabetic retinopathy screening program in the Netherlands. The dataset contained 40 color fundus images, among which 33 images were negative for diabetic retinopathy and seven images were diagnosed with diabetic retinopathy. The images were acquired using a Canon CR5 non-mydriatic 23CCD camera with a 45-degree field of view (FOV). When released to the public, the images were cropped to only include the FOV. The retinal vessels depicted in the images were manually segmented by an ophthalmologist. The retinal images were randomly sampled to create image “patches” that were 48 × 48 pixels.

2. **Kvasir-SEG**: This dataset was generated by the Vestre Viken Health Trust in Norway and consisted of 1000 GI tract endoscopic images depicting polyps. Certified radiologists outlined the polyps on all the images. Image matrices varied from 720 × 576 to 1920 × 1080. All the images in this dataset contained polyps, and their locations were known. In other words, the algorithm did not need to detect the polyps.

3. **CHASE DB1**: This dataset was collected during cardiovascular health screening of primary school children in three different UK cities. The dataset contains 28 color retina images taken from the left and right eye of 14 pediatric subjects. Each image was annotated by two trained specialists. The fundus images were taken with a Nidek NM-200D handheld fundus camera and processed with a Computer-Assisted Image Analysis of the Retina (CAIAR) program.

4. **ISIC**: This dataset contains 2000 images of cancerous skin lesions collected by the International Skin Imaging Collaboration. Each image contains a lesion diagnosis of either melanoma, nevus, or seborrheic keratosis. An experienced clinician used a semi-automated or manual process to segment the lesions on the images.

## Training Procedure

The Dice similarity coefficient (DSC) was used as the loss function to train Super U-net on both datasets. The Adam optimizer was used with initial learning rates of 0.001, 0.0001, 0.01, and 0.01 for DRIVE, Kvasir-SEG, CHASE DB1, and ISIC databases, respectively. Learning rates were determined empirically based upon a specific task. The training epochs for the DRIVE, CHASE DB1, Kvasir-SEG, and ISIC datasets were 40, 40, 30, and 100, respectively. Due to the limit of GPU memory, the batch size was set at 32 (image patches) for the DRIVE and CHASE DB1 datasets and 4 (images) for the ISIC and Kvasir-SEG datasets. The training procedure stopped if the DSC loss did not improve for 15 continuous epochs. When training the networks on the Kvasir-SEG and ISIC dataset, each image was resized to 512 × 512 pixels by nearest neighbor sampling. When training the networks on the DRIVE and CHASE DB1 dataset, we employed random patch generation. This procedure involves randomly selecting 48 × 48 pixel subsections of the original image for training, which increases the size and diversity of the training data. When testing occurred, sliding window patch generation was used to create predictions. After all patches for a testing image were generated, they were “stitched” together to create the complete segmentation map. Training data was augmented using a collection of geometric and image transformations (e.g., scale, rotation, translation, Gaussian noise, smoothing, and brightness perturbations). Pixels that were predicted at or above 0.5 were classified as the regions of interest during the testing phase. All networks were implemented in Keras TensorFlow and trained on an NVIDIA GeForce Titan XP.

## Model Evaluation

To evaluate the model's performance, the trained model was saved and tested on the validation datasets. The evaluation procedure involved loading and preprocessing the validation datasets, augmenting the data, and then using the trained model to make predictions. The model's performance was assessed using the Dice similarity coefficient (DSC) and other relevant metrics.

## Validating Segmentation Performance on 3-D Images

In the future, we will implement the 3-D version of Super U-Net and validate its segmentation performance on radiological images (e.g., computed tomography (CT) and magnetic resonance imaging (MRI)).

## Exploring Potential of Super U-Net for Other Medical Image Analysis Tasks

We will explore the potential of Super U-Net for other medical image analysis tasks (e.g., classification and registration).

## New Techniques and Methods Implemented

To improve the accuracy of SuperUNET for segmentation tasks, the following techniques have been implemented:

* **Attention mechanisms**: Integrate SE (Squeeze-and-Excitation), CBAM (Convolutional Block Attention Module), or Transformer Blocks to allow the model to focus on the most relevant regions.
* **Residual connections**: Use ResUNet-like skip connections to prevent vanishing gradients and enhance feature propagation.
* **Deformable convolutions**: Replace standard convolutions with deformable convolutions to capture complex patterns.
* **Elastic deformations, affine transformations, and CutMix**: These help the model generalize better.
* **MixUp augmentation**: Blends two images and labels for better robustness.
* **Focal loss + Dice loss**: Focal loss improves segmentation on hard-to-classify pixels, while Dice loss balances foreground and background segmentation.
* **Boundary loss**: Helps in better contour segmentation, especially for medical images.
* **Pretraining with large datasets**: Use pretrained encoders like EfficientNet, Swin Transformer, or ConvNeXt for better feature extraction.
* **Progressive learning**: Train with lower resolution first, then gradually increase image size to stabilize learning.
* **Adaptive learning rate scheduling**: Use Cosine Annealing or Cyclical LR for smoother convergence.
* **CRF (Conditional Random Fields)**: Helps refine segmentation boundaries.
* **Test-time augmentation (TTA)**: Apply multiple transformations during inference and average predictions.
* **Ensemble learning**: Combine multiple models for robust segmentation.

## Instructions for Using the New Features and Techniques

1. **Attention Mechanisms**: The attention mechanisms such as SE, CBAM, or Transformer Blocks have been integrated into the model architecture. You can enable these mechanisms by modifying the model definition in the `super_unet.ipynb` notebook.

2. **MixUp Augmentation**: MixUp augmentation has been implemented in the data augmentation section of the `super_unet.ipynb` notebook. You can apply MixUp augmentation by calling the `mixup_data` function on your training data.

3. **Focal Loss and Boundary Loss**: The Focal Loss and Boundary Loss functions have been added to the `super_unet.ipynb` notebook. You can use these loss functions in model compilation by specifying them in the `model.compile` method.

4. **Pretraining with Large Datasets**: Pretrained encoders like EfficientNet, Swin Transformer, or ConvNeXt can be used for better feature extraction. You can load these pretrained encoders in the `super_unet.ipynb` notebook and use them in the model architecture.

5. **Progressive Learning and Adaptive Learning Rate Scheduling**: Progressive learning and adaptive learning rate scheduling have been added to the training section of the `super_unet.ipynb` notebook. You can implement these techniques by following the provided code in the notebook.

6. **CRF, TTA, and Ensemble Learning**: CRF, TTA, and ensemble learning have been implemented in the post-processing section of the `super_unet.ipynb` notebook. You can apply these techniques by calling the respective functions on your model predictions.

By following these instructions, you can utilize the new features and techniques to improve the accuracy and robustness of the SuperUNET model for segmentation tasks.
