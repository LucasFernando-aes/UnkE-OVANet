from .mydataset import ImageFolder, Preprocess
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_data(
        data_path,
        preprocess=None,
        transform=None,
        return_paths=False,
        return_id=False):
    data = ImageFolder(
        data_path,
        transform=preprocess if preprocess is not None else transform,
        return_paths=return_paths,
        return_id=return_id)
    if preprocess is not None:
        return Preprocess(data, transform=transform)
    else:
        return data


def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False, val=False, val_data=None):
    source_folder = get_data(os.path.join(source_path),
                             preprocess=transforms.get("preprocess"),
                             transform=transforms[source_path],
                             return_id=return_id)
    target_folder_train = get_data(os.path.join(target_path),
                                   preprocess=transforms.get("preprocess"),
                                   transform=transforms[target_path],
                                   return_id=return_id)
    if val:
        source_val_train = get_data(val_data,
                                    preprocess=transforms.get("preprocess"),
                                    transform=transforms[source_path],
                                    return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = get_data(val_data,
                                   preprocess=transforms.get("preprocess"),
                                   transform=transforms[evaluation_path],
                                   return_id=return_id)
    eval_folder_test = get_data(os.path.join(evaluation_path),
                                preprocess=transforms.get("preprocess"),
                                transform=transforms["eval"],
                                return_paths=True)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    if val:
        test_loader_source = torch.utils.data.DataLoader(
            source_val_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader, target_loader, test_loader, target_folder_train


def get_loader_label(source_path, target_path, target_path_label, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = get_data(os.path.join(source_path),
                             preprocess=transforms.get("preprocess"),
                             transform=transforms[source_path],
                             return_id=return_id)
    target_folder_train = get_data(os.path.join(target_path),
                                   preprocess=transforms.get("preprocess"),
                                   transform=transforms[target_path],
                                   return_id=return_id)
    target_folder_label = get_data(os.path.join(target_path_label),
                                   preprocess=transforms.get("preprocess"),
                                   transform=transforms[target_path],
                                   return_id=return_id)
    eval_folder_test = get_data(os.path.join(evaluation_path),
                                preprocess=transforms.get("preprocess"),
                                transform=transforms[evaluation_path],
                                return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_loader_label = torch.utils.data.DataLoader(
        target_folder_label,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_loader_label, test_loader, target_folder_train



