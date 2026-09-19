def _generate_hand_skeleton(start_idx: int) -> list:
    conn = []
    conn.extend([
        (start_idx + 0, start_idx + 1), (start_idx + 0, start_idx + 5),
        (start_idx + 0, start_idx + 9), (start_idx + 0, start_idx + 13),
        (start_idx + 0, start_idx + 17)
    ])
    fingers = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for f in fingers:
        for i in range(len(f) - 1):
            conn.append((start_idx + f[i], start_idx + f[i+1]))
    return conn

SKELETON_FORMATS = {
    "coco_wholebody_133": {
        "num_keypoints": 133,
        "parts": {
            "body": {
                "indices": list(range(0, 17)),
                "skeleton": [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
                "color": (0, 255, 0),      # Green
                "radius": 4,
                "thickness": 2
            },
            "foot": {
                "indices": list(range(17, 23)),
                "skeleton": [(17, 19), (19, 21), (18, 20), (20, 22)],
                "color": (128, 255, 0),    # Light green
                "radius": 3,
                "thickness": 2
            },
            "face": {
                "indices": list(range(23, 91)),
                "skeleton": [(i, i+1) for i in range(23, 90)],
                "color": (255, 0, 255),    # Magenta
                "radius": 2,
                "thickness": 1
            },
            "left_hand": {
                "indices": list(range(91, 112)),
                "skeleton": _generate_hand_skeleton(91),
                "color": (255, 255, 0),    # Yellow (Labeled Cyan in original)
                "radius": 3,
                "thickness": 2
            },
            "right_hand": {
                "indices": list(range(112, 133)),
                "skeleton": _generate_hand_skeleton(112),
                "color": (0, 255, 255),    # Cyan (Labeled Yellow in original)
                "radius": 3,
                "thickness": 2
            }
        }
    },
    "coco_17": {
        "num_keypoints": 17,
        "parts": {
            "head": {
                "indices": [0, 1, 2, 3, 4],
                "skeleton": [(0, 1), (0, 2), (1, 3), (2, 4)],
                "color": (255, 0, 255),    # Magenta
                "radius": 4,
                "thickness": 2
            },
            "torso": {
                "indices": [5, 6, 11, 12],
                "skeleton": [(5, 6), (5, 11), (6, 12), (11, 12)],
                "color": (0, 255, 0),      # Green
                "radius": 4,
                "thickness": 2
            },
            "left_arm": {
                "indices": [5, 7, 9],
                "skeleton": [(5, 7), (7, 9)],
                "color": (255, 255, 0),    # Yellow
                "radius": 3,
                "thickness": 2
            },
            "right_arm": {
                "indices": [6, 8, 10],
                "skeleton": [(6, 8), (8, 10)],
                "color": (0, 255, 255),    # Cyan
                "radius": 3,
                "thickness": 2
            },
            "left_leg": {
                "indices": [11, 13, 15],
                "skeleton": [(11, 13), (13, 15)],
                "color": (255, 128, 0),    # Orange
                "radius": 3,
                "thickness": 2
            },
            "right_leg": {
                "indices": [12, 14, 16],
                "skeleton": [(12, 14), (14, 16)],
                "color": (0, 128, 255),    # Blue
                "radius": 3,
                "thickness": 2
            }
        }
    },
    "mediapipe_33": {
        "num_keypoints": 33,
        "parts": {
            "head": {
                "indices": list(range(0, 9)),
                "skeleton": [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)],
                "color": (255, 0, 255),    # Magenta
                "radius": 3,
                "thickness": 2
            },
            "mouth": {
                "indices": [9, 10],
                "skeleton": [(9, 10)],
                "color": (0, 0, 255),      # Red
                "radius": 3,
                "thickness": 2
            },
            "torso": {
                "indices": [11, 12, 23, 24],
                "skeleton": [(11, 12), (11, 23), (12, 24), (23, 24)],
                "color": (0, 255, 0),      # Green
                "radius": 4,
                "thickness": 2
            },
            "left_arm": {
                "indices": [11, 13, 15],
                "skeleton": [(11, 13), (13, 15)],
                "color": (255, 255, 0),    # Yellow
                "radius": 3,
                "thickness": 2
            },
            "left_hand": {
                "indices": [15, 17, 19, 21],
                "skeleton": [(15, 17), (15, 19), (15, 21), (17, 19)],
                "color": (255, 165, 0),    # Orange
                "radius": 3,
                "thickness": 2
            },
            "right_arm": {
                "indices": [12, 14, 16],
                "skeleton": [(12, 14), (14, 16)],
                "color": (0, 255, 255),    # Cyan
                "radius": 3,
                "thickness": 2
            },
            "right_hand": {
                "indices": [16, 18, 20, 22],
                "skeleton": [(16, 18), (16, 20), (16, 22), (18, 20)],
                "color": (0, 165, 255),    # Light Blue
                "radius": 3,
                "thickness": 2
            },
            "left_leg": {
                "indices": [23, 25, 27],
                "skeleton": [(23, 25), (25, 27)],
                "color": (255, 128, 0),    # Dark Orange
                "radius": 3,
                "thickness": 2
            },
            "left_foot": {
                "indices": [27, 29, 31],
                "skeleton": [(27, 29), (27, 31), (29, 31)],
                "color": (128, 0, 255),    # Purple
                "radius": 3,
                "thickness": 2
            },
            "right_leg": {
                "indices": [24, 26, 28],
                "skeleton": [(24, 26), (26, 28)],
                "color": (0, 128, 255),    # Blue
                "radius": 3,
                "thickness": 2
            },
            "right_foot": {
                "indices": [28, 30, 32],
                "skeleton": [(28, 30), (28, 32), (30, 32)],
                "color": (255, 0, 128),    # Pink
                "radius": 3,
                "thickness": 2
            }
        }
    },
    "openpose_25": {
        "num_keypoints": 25,
        "parts": {
            "head": {
                "indices": [0, 14, 15, 16, 17],
                "skeleton": [(0, 1), (0, 14), (14, 16), (0, 15), (15, 17)],
                "color": (255, 0, 255),    # Magenta
                "radius": 4,
                "thickness": 2
            },
            "torso": {
                "indices": [1, 8, 11],
                "skeleton": [(1, 8), (1, 11)],
                "color": (0, 255, 0),      # Green
                "radius": 4,
                "thickness": 2
            },
            "right_arm": {
                "indices": [2, 3, 4],
                "skeleton": [(1, 2), (2, 3), (3, 4)],
                "color": (255, 255, 0),    # Yellow
                "radius": 3,
                "thickness": 2
            },
            "left_arm": {
                "indices": [5, 6, 7],
                "skeleton": [(1, 5), (5, 6), (6, 7)],
                "color": (0, 255, 255),    # Cyan
                "radius": 3,
                "thickness": 2
            },
            "right_leg": {
                "indices": [8, 9, 10],
                "skeleton": [(8, 9), (9, 10)],
                "color": (255, 128, 0),    # Orange
                "radius": 3,
                "thickness": 2
            },
            "left_leg": {
                "indices": [11, 12, 13],
                "skeleton": [(11, 12), (12, 13)],
                "color": (0, 128, 255),    # Blue
                "radius": 3,
                "thickness": 2
            },
            "feet": {
                "indices": [19, 20, 21, 22, 23, 24],
                "skeleton": [(10, 22), (13, 19), (22, 23), (23, 24), (19, 20), (20, 21)],
                "color": (128, 0, 255),    # Purple
                "radius": 3,
                "thickness": 2
            }
        }
    }
}
