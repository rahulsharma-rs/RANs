from PIL import Image

def crop(image_path, coords, saved_location):

    """

    @param image_path: The path to the image to edit

    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)

    @param saved_location: Path to save the cropped image

    """

    image_obj = Image.open(image_path)

    cropped_image = image_obj.crop(coords)

    cropped_image.save(saved_location)

    cropped_image.show()

if __name__ == '__main__':

    image = '/home/rahul/Dropbox/RAHUL_PhD_Project/PhD/draft_papers/Journal/ijns-2e/ober-exp4.png'

    crop(image, (25, 35, 620, 590), './ober-exp4.png')