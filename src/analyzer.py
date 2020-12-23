import sys, json
from torchvision import datasets, models, transforms
from PIL import Image

def infer(frame):
    """Need a comment"""

    with open("imagenet-simple-labels.json") as f:
        labels = json.load(f)

    # TODO: Need to get the right transfor sizes for each model I'm going to use
    data_transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])

    # Should we normalize?
    # https://pytorch.org/docs/stable/torchvision/models.html#video-classification

    # Load the image
    #image = Image.open(frame)
    image = Image.fromarray(frame)
    # Apply the transformation, expand the batch dimension, and send the image to the GPU
    image = data_transform(image).unsqueeze(0).cuda()

    # Download the model if it's not there already. It will take a bit on the first run, after that it's fast
    mobilenet = models.mobilenet_v2(pretrained=True)
    # Send the model to the GPU
    mobilenet.cuda()
    # Set layers such as dropout and batchnorm in evaluation mode
    mobilenet.eval()

    # Get the 1000-dimensional model output
    out = mobilenet(image)
    # Find the predicted class
    res = "Predicted class is: {}".format(labels[out.argmax()])
    return res

    # TODO: Draw a bounding box over the image with the predicted result.
    # TODO : Return the results in this function

if __name__ == "__main__":
    test_image = '../test/alaskan-malamute.jpg'
    test = infer(test_image)
    print(test)