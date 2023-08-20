import torchvision
from torchvision import transforms
import torchvision.datasets as datasets


# CIFAR10
CIFAR10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

CIFAR10_transform_test = transforms.ToTensor()

CIFAR10_means, CIFAR10_stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

CIFAR10_training_set = torchvision.datasets.CIFAR10(root = f'./data', train = True, transform = CIFAR10_transform_train, download = True)
CIFAR10_testing_set = torchvision.datasets.CIFAR10(root = f'./data', train = False, transform = CIFAR10_transform_test, download = True)
CIFAR10_dataset = torchvision.datasets.CIFAR10(root = f'./data', train = False, transform = None, download = True)
CIFAR10_num_classes = 10


# GTSRB
GTSRB_transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

GTSRB_transform_test = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ToTensor()
])

GTSRB_means, GTSRB_stds = (0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)

GTSRB_training_set = torchvision.datasets.GTSRB(root = f'./data', split = 'train', transform = GTSRB_transform_train, download = True)
GTSRB_testing_set = torchvision.datasets.GTSRB(root = f'./data', split = 'test', transform = GTSRB_transform_test, download = True)
GTSRB_dataset = torchvision.datasets.GTSRB(root = f'./data', split = 'train', transform = None, download = True)
GTSRB_num_classes = 43

# TINY
tiny_transform_train = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])
tiny_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
tiny_means, tiny_stds = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
tiny_training_set = datasets.ImageFolder('./data/tiny-imagenet-200/train', tiny_transform_train)
tiny_testing_set = datasets.ImageFolder('./data/tiny-imagenet-200/test', tiny_transform_test)
tiny_num_classes = 200
