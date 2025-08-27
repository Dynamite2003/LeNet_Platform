import cv2
import torch


def display_image(img, target_size=None):
    if target_size:
        ratio = target_size / max(img.shape[0], img.shape[1])
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=interp)
    cv2.imshow("", img)
    cv2.waitKey(0)


def demo_display_single_image(images, labels):
    image_to_display = images[0]
    image_to_display = image_to_display.numpy()
    image_to_display = image_to_display.transpose(1, 2, 0)
    display_image(image_to_display, target_size=240)
    label = labels[0]
    cv2.imwrite(f"{label.item}.jpg", image_to_display)


def demo_display_specific_digit_combination(images, labels):
    assert images.shape[0] == labels.shape[0]

    # this is an example for retrieve digits image
    digit_image_dict = {}
    for i in range(0, labels.shape[0]):
        digit = int(labels[i].item())
        if digit not in digit_image_dict:
            digit_image_dict[digit] = images[i]

        if len(digit_image_dict.keys()) >= 10:
            break

    ##########  CODE START  ##########
  
    my_student_number = "2022011283"
    number_image_list = [digit_image_dict[int(d)] for d in my_student_number]
    # 沿宽度方向拼接
    target_img = torch.concat(number_image_list,dim=2)

    ##########  CODE END  ##########

    target_img = target_img.numpy()
    target_img = target_img.transpose(1, 2, 0)
    display_image(target_img, target_size=240)


