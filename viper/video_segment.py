from __future__ import annotations

import torch
from typing import Union, Iterator, Callable, Dict

from .image_patch import ImagePatch


class VideoSegment:
    """A Python class containing a set of frames represented as ImagePatch objects, as well as relevant information.
    Attributes
    ----------
    video : torch.Tensor
        A tensor of the original video.
    start : int
        An int describing the starting frame in this video segment with respect to the original video.
    end : int
        An int describing the ending frame in this video segment with respect to the original video.
    num_frames->int
        An int containing the number of frames in the video segment.

    Methods
    -------
    frame_iterator->Iterator[ImagePatch]
    trim(start, end)->VideoSegment
        Returns a new VideoSegment containing a trimmed version of the original video at the [start, end] segment.
    """

    def __init__(self, config: Dict, forward: Callable, video: torch.Tensor, start: int = None, end: int = None, parent_start=0):
        """Initializes a VideoSegment object by trimming the video at the given [start, end] times and stores the
        start and end times as attributes. If no times are provided, the video is left unmodified, and the times are
        set to the beginning and end of the video.

        Parameters
        -------
        video : torch.Tensor
            A tensor of the original video.
        start : int
            An int describing the starting frame in this video segment with respect to the original video.
        end : int
            An int describing the ending frame in this video segment with respect to the original video.
        """

        self.config = config
        self.forward = forward

        if start is None and end is None:
            self.trimmed_video = video
            self.start = 0
            self.end = video.shape[0]  # duration
        else:
            self.trimmed_video = video[start:end]
            if start is None:
                start = 0
            if end is None:
                end = video.shape[0]
            self.start = start + parent_start
            self.end = end + parent_start

        self.num_frames = self.trimmed_video.shape[0]

        self.cache = {}

        if self.trimmed_video.shape[0] == 0:
            raise Exception("VideoSegment has duration=0")

    def frame_from_index(self, index) -> ImagePatch:
        """Returns the frame at position 'index', as an ImagePatch object."""
        if index < self.num_frames:
            image = self.trimmed_video[index]
        else:
            image = self.trimmed_video[-1]
        return ImagePatch(self.config, self.forward, image)

    def trim(self, start: Union[int, None] = None, end: Union[int, None] = None) -> VideoSegment:
        """Returns a new VideoSegment containing a trimmed version of the original video at the [start, end]
        segment.

        Parameters
        ----------
        start : Union[int, None]
            An int describing the starting frame in this video segment with respect to the original video.
        end : Union[int, None]
            An int describing the ending frame in this video segment with respect to the original video.

        Returns
        -------
        VideoSegment
            a new VideoSegment containing a trimmed version of the original video at the [start, end]
        """
        if start is not None:
            start = max(start, 0)
        if end is not None:
            end = min(end, self.num_frames)

        return VideoSegment(self.config, self.forward, self.trimmed_video, start, end, self.start)

    def frame_iterator(self) -> Iterator[ImagePatch]:
        """Returns an iterator over the frames in the video segment."""
        for i in range(self.num_frames):
            yield ImagePatch(self.config, self.forward, self.trimmed_video[i])

    def __repr__(self):
        return "VideoSegment({}, {})".format(self.start, self.end)
