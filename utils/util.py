import torch

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):
        # i번째 프레임의 video feature를 win_size만큼 repeat(왜 이렇게 하는지 모르겠음.)
        # i ~ i + win_size에 해당하는 구간에 대해서 distance를 계산함
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :]
            )
        )

    return dists
