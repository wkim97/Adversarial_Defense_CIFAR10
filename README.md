# Adversarial_Defense_CIFAR10

### Adversarial Defense on CIFAR10 dataset

### Codeflow:
1. **clean_model_training/train_clean.py**:
    * Train ResNet18 model on clean CIFAR10 training dataset.
    * Resulting model saved in **models/CIFAR10_net.pth**
    * Accuracy results saved in **results/accuracy_on_clean_model.csv**
2. **noise_modeling/apply_noise.py**:
    * Applies PGD noise on CIFAR10 training dataset
    * Saves resulting images to **data/noisy_images**
3. **noise_modeling/noise_extraction.py**:
    * Extract noise from noisy images from **data/noisy_images**
    * Store resulting noise to **data/noise/noise** and **data/noise/denoised**
4. **noise_modeling/gan_model.py** and **noise_modeling/train_gan.py**:
    * **gan_model.py** - defines generator and discriminator for DCGAN
    * **train_gan.py** - trains the GAN model with noise samples from **data/noise/noise**
    * Saves the resulting noise models to **data/noise_models**
    * Saves the resulting discriminator and generator to **models/gan_models**
5. **noise_modeling/add_noise_to_clean.py**:
    * Adds noise from **data/noise_models** to clean training images of CIFAR10
    * Stores the resulting noisy images to **data/noisy_CIFAR10**
6. **denoising/DnCNN_model.py** and **denoising/DnCNN_train.py**:
    * Train the DnCNN denoising model with noisy dataset and clean dataset from **data/noisy_CIFAR10/train/generated_noisy_images**
    and **data/noisy_CIFAR10/train/clean_images**
    * Stores the resulting model as **models/DnCNN_model.pth**
7. **denoising/DnCNN_test.py**:
    * Test trained DnCNN denoising model with CIFAR10 test images attacked by various adversarial attacks.
    * Results stored as csv files in **results**