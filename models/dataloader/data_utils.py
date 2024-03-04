from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniimagenet':
        from models.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        from models.dataloader.cub_cropped import CUB as Dataset
    elif args.dataset == 'tieredimagenet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'dogs':
        from models.dataloader.dogs import Dogs as Dataset
    elif args.dataset == 'cars':
        from models.dataloader.cars import Cars as Dataset
    elif args.dataset == 'flowers':
        from models.dataloader.flowers import Flowers as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
