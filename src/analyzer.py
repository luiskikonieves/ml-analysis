import sys, json
import torch
from torchvision import datasets, models, transforms, utils
from PIL import Image


def imagenet(frame):
    """Detects the objects in an image frame using pre-trained Imagenet v2.

    :param array frame: an image frame
    :return array res: an array with the results of the detection:
    """

    with open("imagenet-classes-simple.json") as f:
        labels = json.load(f)

    # TODO: Need to get the right transform sizes for each model I'm going to use
    # Should we normalize?
    # https://pytorch.org/docs/stable/torchvision/models.html#video-classification
    data_transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])

    # Load the image as an array, otherwise tf is going to freak that it's not a np array
    image = Image.open(frame)
    #image = Image.fromarray(frame)
    # Apply the transformation, expand the batch dimension, and send the image to the GPU
    image = data_transform(image).unsqueeze(0).cuda()

    # Download the model if it's not there already. It will take a bit on the first run, after that it's fast
    mobilenet = models.mobilenet_v2(pretrained=True)
    # Send the model to the GPU
    mobilenet.cuda()
    # Set layers such as dropout and batchnorm in evaluation mode
    mobilenet.eval()

    # Get the 1000-dimensional model output
    #with torch.no_grad():
    #    detections = mobilenet(image)
        #detections = utils.non_max_supression(detections, 80, conf_thres, nms_thres)
    #return detections[0]

    out = mobilenet(image)
    # Find the predicted class
    res = "Predicted class is: {}".format(labels[out.argmax()])
    return res

    # TODO: Draw a bounding box over the image with the predicted result.
    test = utils.d


if __name__ == "__main__":
    test_image = '../test/tiger-and-dolphin.jpg'
    test = imagenet(test_image)
    print(test)
