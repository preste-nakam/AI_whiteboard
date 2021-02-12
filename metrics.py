def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou


def get_stat(gt, pr):
    n = len(gt)
    smth = 0.00000001
    tp,tn,fp,fn = 0,0,0,0
    for i in range(n):
        l1, l2 = gt[i], pr[i]
        if (l1 == 1) and (l2 == 1):
            tp +=1
        elif (l1 == 1) and (l2 == 0):
            fp += 1
        elif (l1 == 0) and (l2 == 1):
            fn += 1
        elif (l1 == 0) and (l2 == 0):
            tn += 1

    print('\ntp - {},tn - {},fp - {},fn - {}\n'.format(tp,tn,fp,fn))
    acc = (tp + tn) / (tp + tn + fp + fn+smth)
    recall = tp / (tp + fp + smth)
    precision = tp / (tp + fn + smth)
    
    return acc, recall, precision, [tp,tn,fp,fn]