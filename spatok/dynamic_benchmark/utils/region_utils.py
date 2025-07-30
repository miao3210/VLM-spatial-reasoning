import pdb 
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from .seg_match_and_redundancy_utils import find_matched_segment, summarize_overlap_segments
from .basic_math_utils import cluster_related_features


def extract_region_between_segments(img, seg1=None, seg2=None, segs=None, straighten=False, dilation=20):
    # def crop_between_segments(seg1, seg2, img, straighten=False):
    """
    Extract the region between two parallel segments from an image.

    Parameters:
        seg1, seg2: np.array([x1, y1, x2, y2])
        img: input image (BGR)
        straighten: if True, un-rotate the region using perspective transform

    Returns:
        Cropped region (possibly rotated upright if straighten=True)
    """
    if seg2 is None and segs is None:
        raise ValueError("Either seg2 or segs must be provided.")
    
    if seg1 is not None:
        p1 = seg1[:2]
        p2 = seg1[2:]

        if seg2 is not None and segs is None:
            # Get endpoints
            p3 = seg2[:2]
            p4 = seg2[2:]
            # Define quadrilateral region (clockwise or counterclockwise)
            d = np.dot(p2 - p1, p4 - p3)
            if d > 0:
                region = np.array([p1, p2, p4, p3], dtype=np.int32)
            elif d < 0:
                region = np.array([p1, p2, p3, p4], dtype=np.int32)
            else:
                raise ValueError("Segments are collinear.")
            if not straighten:
                # Create mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [region], 255)
                # Thick the mask by 10 pixels
                mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))

                # Apply mask
                masked = cv2.bitwise_and(img, img, mask=mask)

                # Crop bounding box around polygon
                x, y, w, h = cv2.boundingRect(region)
                cropped = masked[y:y+h, x:x+w]
                return cropped
            else:
                raise NotImplementedError("Current perspective transform does not work well.")
                # Straighten the region using perspective transform
                width = int(np.linalg.norm(p1 - p2))
                height = int(np.linalg.norm(p1 - p3))
                dst_pts = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                src_pts = np.array([p1, p2, p4, p3], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                # Warp perspective to extract upright region
                warped = cv2.warpPerspective(img, M, (width, height))
                return warped
            
        elif segs is not None and seg2 is None:
            mask_large = np.zeros(img.shape[:2], dtype=np.uint8)

            if not straighten:
                for seg2 in segs:
                    # Get endpoints
                    p3 = seg2[:2]
                    p4 = seg2[2:]
                    # Define quadrilateral region (clockwise or counterclockwise)
                    d = np.dot(p2 - p1, p4 - p3)
                    if d > 0:
                        region = np.array([p1, p2, p4, p3], dtype=np.int32)
                    elif d < 0:
                        region = np.array([p1, p2, p3, p4], dtype=np.int32)
                    else:
                        raise ValueError("Segments are collinear.")
                    
                    # Create mask
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [region], 255)
                    # Thick the mask by 10 pixels
                    mask = cv2.dilate(mask, np.ones((20, 20), np.uint8))
                    mask_large = cv2.bitwise_or(mask_large, mask)

                # Apply mask
                masked = cv2.bitwise_and(img, img, mask=mask_large)

                # Crop bounding box around polygon
                x, y, w, h = cv2.boundingRect(mask_large)
                cropped = masked[y:y+h, x:x+w]

                return cropped
            
            else:
                raise NotImplementedError("Current perspective transform does not work well.")
        
        else:
            raise ValueError("When seg1 is given, only one of seg2 or segs can be provided.")
        
    else:
        if segs is None:
            raise ValueError("When seg1 is not provided, at least one segment must be provided for segs.")
        if len(segs) == 0:
            raise ValueError("No segments provided in segs.")
        elif len(segs) == 1:
            raise ValueError("Only one segment provided in segs.")
        else:
            mask_large = np.zeros(img.shape[:2], dtype=np.uint8)
            for k1, seg1 in enumerate(segs):
                p1 = seg1[:2]
                p2 = seg1[2:]
                for k2, seg2 in enumerate(segs):
                    p3 = seg2[:2]
                    p4 = seg2[2:]
                    # Define quadrilateral region (clockwise or counterclockwise)
                    d = np.dot(p2 - p1, p4 - p3)
                    if d > 0:
                        region = np.array([p1, p2, p4, p3], dtype=np.int32)
                    elif d < 0:
                        region = np.array([p1, p2, p3, p4], dtype=np.int32)
                    else:
                        raise ValueError("Segments are collinear.")
                    
                    # Create mask
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [region], 255)
                    # Thick the mask by 10 pixels
                    mask = cv2.dilate(mask, np.ones((20, 20), np.uint8))
                    mask_large = cv2.bitwise_or(mask_large, mask)

            masked = cv2.bitwise_and(img, img, mask=mask_large)

            # Apply mask
            masked = cv2.bitwise_and(img, img, mask=mask_large)

            # Crop bounding box around polygon
            x, y, w, h = cv2.boundingRect(mask_large)
            cropped = masked[y:y+h, x:x+w]

            return cropped



def extract_region_around_point(imgsize, points, dilation=200):
    h, w = imgsize[:2]
    # 1) seed mask in 0/255
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(points).reshape(-1, 2)
    for point in points:
        c, r = point
        if 0 <= r < h and 0 <= c < w:
            seed_mask[int(r), int(c)] = 255

    # 2) circular kernel
    k = 2*dilation + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 3) dilate — stays in 0/255
    mask = cv2.dilate(seed_mask, kernel)
    return mask  # dtype=uint8, values={0,255}




def extract_mask_of_region_between_segments(img_shape, seg1=None, seg2=None, segs=None, dilation=20):
    # def crop_between_segments(seg1, seg2, img, straighten=False):
    """
    Extract the region between two parallel segments from an image.

    Parameters:
        seg1, seg2: np.array([x1, y1, x2, y2])
        img: input image (BGR)
        straighten: if True, un-rotate the region using perspective transform

    Returns:
        Cropped region (possibly rotated upright if straighten=True)
    """
    if seg2 is None and segs is None:
        raise ValueError("Either seg2 or segs must be provided.")
    
    if seg1 is not None:
        p1 = seg1[:2]
        p2 = seg1[2:]

        if seg2 is not None and segs is None:
            # Get endpoints
            p3 = seg2[:2]
            p4 = seg2[2:]
            # Define quadrilateral region (clockwise or counterclockwise)
            d = np.dot(p2 - p1, p4 - p3)
            if d > 0:
                region = np.array([p1, p2, p4, p3], dtype=np.int32)
            elif d < 0:
                region = np.array([p1, p2, p3, p4], dtype=np.int32)
            else:
                raise ValueError("Segments are collinear.")
            # Create mask
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)
            # Thick the mask by 10 pixels
            mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
            
            return mask
            
        elif segs is not None and seg2 is None:
            mask_large = np.zeros(img_shape[:2], dtype=np.uint8)

            for seg2 in segs:
                # Get endpoints
                p3 = seg2[:2]
                p4 = seg2[2:]
                # Define quadrilateral region (clockwise or counterclockwise)
                d = np.dot(p2 - p1, p4 - p3)
                if d > 0:
                    region = np.array([p1, p2, p4, p3], dtype=np.int32)
                elif d < 0:
                    region = np.array([p1, p2, p3, p4], dtype=np.int32)
                else:
                    raise ValueError("Segments are collinear.")
                
                # Create mask
                mask = np.zeros(img_shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [region], 255)
                # Thick the mask by 10 pixels
                mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
                mask_large = cv2.bitwise_or(mask_large, mask)
            
            return mask_large
            
        else:
            raise ValueError("When seg1 is given, only one of seg2 or segs can be provided.")
        
    else:
        if segs is None:
            raise ValueError("When seg1 is not provided, at least two segment must be provided for segs.")
        if len(segs) == 0:
            raise ValueError("No segments provided in segs. When seg1 is not provided, at least two segment must be provided for segs.")
        elif len(segs) == 1:
            offsets = [
                [dilation, dilation],
                [-dilation, -dilation],
                [dilation, -dilation],
                [-dilation, dilation]
            ]
            new_segs = [segs]
            for offset in offsets:
                new_seg = (segs.reshape(2,2) + np.array(offset).reshape(1,2)).reshape(4)
                new_segs.append(new_seg)
        
        mask_large = np.zeros(img_shape[:2], dtype=np.uint8)
        for k1, seg1 in enumerate(segs):
            p1 = seg1[:2]
            p2 = seg1[2:]
            for k2, seg2 in enumerate(segs):
                p3 = seg2[:2]
                p4 = seg2[2:]
                # Define quadrilateral region (clockwise or counterclockwise)
                d = np.dot(p2 - p1, p4 - p3)
                if d > 0:
                    region = np.array([p1, p2, p4, p3], dtype=np.int32)
                elif d < 0:
                    region = np.array([p1, p2, p3, p4], dtype=np.int32)
                else:
                    raise ValueError("Segments are collinear.")
                
                # Create mask
                mask = np.zeros(img_shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [region], 255)
                # Thick the mask by 10 pixels
                mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
                mask_large = cv2.bitwise_or(mask_large, mask)
        
            return mask_large
        
