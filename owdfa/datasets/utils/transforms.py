import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def create_data_transforms(args, split='train'):
    image_size = getattr(args, 'image_size', 224)
    mean = getattr(args, 'mean', [0.485, 0.456, 0.406])
    std = getattr(args, 'std', [0.229, 0.224, 0.225])
    num_segments = args.num_segments if 'num_segments' in args else 1

    additional_targets = {}
    for i in range(1, num_segments):
        additional_targets[f'image{i}'] = 'image'

    if split == 'train':
        transfrom = alb.Compose([
            alb.HorizontalFlip(),
            alb.RandomResizedCrop(int((256 / 224) * image_size),
                                  int((256 / 224) * image_size), scale=(0.5, 1.0), p=0.2),
            alb.RandomBrightnessContrast(p=0.2),
            alb.Resize(image_size, image_size),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    else:
        transfrom = alb.Compose([
            alb.Resize(image_size, image_size),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return transfrom
