from torchvision import transforms
import torchvision.datasets as datasets


# CIFAR10
class CIFAR10:
    def __init__(self):
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        transform_test = transforms.ToTensor()

        self.C, self.H, self.W = 3, 32, 32
        self.means, self.stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        self.training_set = datasets.CIFAR10(root = f'./data', train = True, transform = transform_train, download = True)
        self.testing_set = datasets.CIFAR10(root = f'./data', train = False, transform = transform_test, download = True)
        self.num_classes = 10
        self.dataset = datasets.CIFAR10(root = f'./data', train = False, transform = None, download = True)

# GTSRB
class GTSRB:
    def __init__(self):

        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        self.means, self.stds = (0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)
        self.C, self.H, self.W = 3, 32, 32
        self.training_set = datasets.GTSRB(root = f'./data', split = 'train', transform = transform_train, download = True)
        self.testing_set = datasets.GTSRB(root = f'./data', split = 'test', transform = transform_test, download = True)
        self.dataset = datasets.GTSRB(root = f'./data', split = 'train', transform = None, download = True)
        self.num_classes = 43

# TINY
class TINY:
    def __init__(self):

        transform_train = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

        self.C, self.H, self.W = 3, 64, 64
        self.means, self.stds = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        self.training_set = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform_train)
        self.testing_set = datasets.ImageFolder('./data/tiny-imagenet-200/test', transform_test)
        self.num_classes = 200
