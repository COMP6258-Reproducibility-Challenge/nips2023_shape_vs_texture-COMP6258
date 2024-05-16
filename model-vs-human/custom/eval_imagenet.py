import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image


# Function to evaluate the model
def evaluate_model(model, image_folder, label_file, num_images):
    model.eval()
    correct = 0
    total = 0

    # Read the label file
    with open(label_file, 'r') as file:
        lines = file.readlines()[:num_images]
        labels = [int(line.strip().split()[1]) for line in lines]

    # Iterate over each image in the image folder
    image_files = sorted(os.listdir(image_folder))[:num_images]
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(image_folder, filename)

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        image_tensor = input_tensor.unsqueeze(0)

        # Perform inference
        # outputs = model(image_tensor)
        with torch.no_grad():
            outputs = model(image_tensor)

        _, predicted = torch.max(outputs.data, 1)

        # probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # print(f"max probability: {max(probabilities)}")
        # print(f"max index: {torch.argmax(probabilities)}")

        # print(f"predicted: {int(predicted)}" )
        # print(f"actual:{labels[idx]}")

        # Check if prediction matches the true label
        total += 1
        if int(predicted) == labels[idx]:
            correct += 1

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {:.2f} %'.format(accuracy))
    # acc.append(accuracy)