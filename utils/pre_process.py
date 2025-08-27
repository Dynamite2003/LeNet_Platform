import torchvision


def normal_transform():
    normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return normal

def data_augment_transform():
    data_augment = torchvision.transforms.Compose([
        # TODO: add more image transforms
        # 随机裁剪 随机旋转 水平翻转
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(28,scale=(0.8,1.0)),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    return data_augment
