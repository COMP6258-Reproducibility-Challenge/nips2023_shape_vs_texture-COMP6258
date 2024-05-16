import torchvision.models as models
import modelling
from eval_imagenet import evaluate_model

# Load the pretrained AlexNet model
alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')

# Paths to image folder and label file
image_folder = "./datasets/imagenet_validation/val_images"
label_file = "./datasets/imagenet_validation/val.txt"


acc = []
num_img = 100

################################################################
print("Original Alexnet")

evaluate_model(alexnet, image_folder, label_file, num_img)
################################################################
print("Alexnet with 5% topK")

alexnet_5 = modelling.AlexNet_topK_5()

evaluate_model(alexnet_5, image_folder, label_file, num_img)
################################################################
print("Alexnet with 10% topK")

alexnet_10 = modelling.AlexNet_topK_10()

evaluate_model(alexnet_10, image_folder, label_file, num_img)
################################################################
print("Alexnet with 20% topK")

alexnet_20 = modelling.AlexNet_topK_20()

evaluate_model(alexnet_20, image_folder, label_file, num_img)
################################################################
print("Alexnet with 30% topK")

alexnet_30 = modelling.AlexNet_topK_30()

evaluate_model(alexnet_30, image_folder, label_file, num_img)
################################################################
print("Alexnet with 40% topK")

alexnet_40 = modelling.AlexNet_topK_40()

evaluate_model(alexnet_40, image_folder, label_file, num_img)
################################################################
print("Alexnet with 50% topK")

alexnet_50 = modelling.AlexNet_topK_50()

evaluate_model(alexnet_50, image_folder, label_file, num_img)
################################################################