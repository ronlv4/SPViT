from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS

from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class ImagenetSubset(VisionDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            subset_file: str = None,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(ImagenetSubset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        with open(subset_file, 'r') as f:
            subset_lines = f.read().splitlines()
        classes = []
        for line in subset_lines:
            class_id, class_name = line.split(' ', 1)
            classes.append(class_id)

        classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print("self.root: ",self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
