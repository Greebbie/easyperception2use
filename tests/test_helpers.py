"""Shared mock classes and helpers for perception pipeline tests."""


# ---------------------------------------------------------------------------
# Mock YOLO result classes (simulate ultralytics tensor-like objects)
# ---------------------------------------------------------------------------

class MockTensor:
    """Simulates a PyTorch tensor with .tolist() and int()/float() conversion."""

    def __init__(self, value):
        self._value = value

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def tolist(self):
        return self._value if isinstance(self._value, list) else [self._value]

    def __getitem__(self, idx):
        if isinstance(self._value, list):
            return MockTensor(self._value[idx])
        return self


class MockBox:
    """Simulates a single YOLO detection box."""

    def __init__(self, track_id, cls_id, conf, xyxy):
        self.id = MockTensor(track_id) if track_id is not None else None
        self.cls = MockTensor(cls_id)
        self.conf = MockTensor(conf)
        # xyxy[0] should return the 4 coords as a list via .tolist()
        self.xyxy = MockTensor([xyxy])


class MockBoxes:
    """Simulates the boxes collection on a YOLO result."""

    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __bool__(self):
        return len(self._boxes) > 0


class MockResult:
    """Simulates a single YOLO result (results[0])."""

    def __init__(self, boxes, names):
        self.boxes = MockBoxes(boxes) if boxes is not None else None
        self.names = names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

YOLO_NAMES = {0: "person", 1: "car", 2: "dog", 3: "chair", 4: "bottle"}


def make_yolo_results(detections, names=None):
    """Build mock YOLO results from a list of detection tuples.

    Each detection: (track_id, cls_name, conf, x1, y1, x2, y2)
    Returns a list with one MockResult (matching YOLO's output format).
    """
    if names is None:
        names = YOLO_NAMES
    name_to_id = {v: k for k, v in names.items()}

    boxes = []
    for det in detections:
        track_id, cls_name, conf, x1, y1, x2, y2 = det
        cls_id = name_to_id.get(cls_name, 0)
        boxes.append(MockBox(track_id, cls_id, conf, [x1, y1, x2, y2]))

    return [MockResult(boxes, names)]
