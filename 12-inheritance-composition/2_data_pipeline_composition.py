from abc import ABC, abstractmethod


class Transform(ABC):
    """Base class for data transforms"""

    @abstractmethod
    def __call__(self, data):
        pass


class Normalize(Transform):
    # TODO: Implement normalization
    pass


class Augment(Transform):
    # TODO: Implement augmentation
    pass


class Compose:
    """Compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        # TODO: Apply all transforms in sequence
        pass


# Much more flexible than inheritance!
pipeline = Compose([Normalize(), Augment(), ...])
