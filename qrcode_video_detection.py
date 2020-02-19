import numpy as np
import cv2
import math
import pyzbar.pyzbar as pyzbar
import json

#from storage_utility import register_qr
#from storage_utility import report

QR_OR_NORTH = 1
QR_OR_EAST = 2
QR_OR_SOUTH = 3
QR_OR_WEST = 4


def resize(img):
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def calculate_distance(p1, p2):
    distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return distance


#p1, p2 - medians, p3 - top point
def calculate_perpendicular_distance(p1, p2, p3):

    #print(p1, p2, p3)

    A = -((p2[1] - p1[1]) / (p2[0] - p1[0]))
    B = 1.0
    C = (((p2[1] - p1[1]) / (p2[0] - p1[0])) * p1[0]) - p1[1]

    return (A * p3[0] + (B * p3[1]) + C) / math.sqrt((A ** 2) + (B ** 2))


def calculate_slope(p1, p2):

    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    return delta_y / delta_x


def calculate_center(p1, p2):
    center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    return center

def find_extreme(center_point, polygon):
    k = 0
    dist = 0

    for i in range(0, 4):
        poly_point = polygon[i]
        c_dist = calculate_distance(center_point, poly_point)
        if c_dist > dist:
            k = i
            dist = c_dist

    return tuple(polygon[k])

def find_minimum(center_point, polygon):
    k = 0
    dist = 100000000

    for i in range(0, 4):
        poly_point = polygon[i]
        c_dist = calculate_distance(center_point, poly_point)
        if c_dist < dist:
            k = i
            dist = c_dist

    return tuple(polygon[k])

def find_intersection(p1, p2, p3, p4):

    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = B1 * p1[1] + A1 * p1[0]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = B2 * p3[1] + A2 * p3[0]

    det = A1*B2 - A2*B1

    if det == 0:
        return (0,0)

    return int((B2*C1 - B1*C2)/det), int((A1*C2 - A2*C1)/det)

def find_contours(img_barcode):
    img_barcode_gs = cv2.cvtColor(img_barcode, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    img_barcode_bin = cv2.adaptiveThreshold(img_barcode_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    contours, hierarchy = cv2.findContours(img_barcode_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def find_pattern_contours(contours, hierarchy):
    hierarchy_valid = []  # hijerarhije validnih kontura
    approximations = {}  # aproksimacije validnih kontura

    frequency_dict = {}  # broj ponavljanja istog roditelja kod hijerarhija kontura
    position_points_polygons = []  # poligoni position patterna kod qr koda

    i = 0
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 3:
                hierarchy_valid.append((i, hierarchy[0][i]))
                approximations[i] = approx

        i = i + 1

    for h in hierarchy_valid:
        if not frequency_dict.get(h[1][3]):
            frequency_dict[h[1][3]] = [h[0]]
        else:
            frequency_dict[h[1][3]].append(h[0])

    for key, value in frequency_dict.items():
        if len(value) == 3:
            for cnt_index in frequency_dict[key]:
                contour_approx = approximations[cnt_index]
                position_points_polygons.append(contour_approx)

    return position_points_polygons

def find_pattern_centers(position_points_polygons):

    central_points = {}
    centers = []

    for position_poly in position_points_polygons:
        coords_1 = position_poly[0][0]
        coords_2 = position_poly[1][0]
        coords_3 = position_poly[2][0]
        coords_4 = position_poly[3][0]

        sum_x = coords_1[0] + coords_2[0] + coords_3[0] + coords_4[0]
        sum_y = coords_1[1] + coords_2[1] + coords_3[1] + coords_4[1]

        center_x = int(sum_x / 4)
        center_y = int(sum_y / 4)

        central_points[(center_x, center_y)] = position_poly
        centers.append((center_x, center_y))

    return central_points, centers

def geometrical_transformation(img, extreme_T, extreme_R, extreme_B, intersection):

    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array([extreme_T, extreme_R, extreme_B, intersection])
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


def process_frame(frame):

    img_barcode = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = img_barcode.copy()

    contours, hierarchy = find_contours(img_barcode)

    position_pattern_polygons = find_pattern_contours(contours, hierarchy)

    if len(position_pattern_polygons) != 3:
        return img, None

    central_points, centers = find_pattern_centers(position_pattern_polygons)

    if len(centers) != 3:
        return img, None

    AB = calculate_distance(centers[0], centers[1])
    BC = calculate_distance(centers[1], centers[2])
    CA = calculate_distance(centers[2], centers[0])

    median_1 = (0,0)
    median_2 = (0,0)
    top = (0,0)
    right = (0,0)
    bottom = (0,0)
    orientation = QR_OR_NORTH

    if AB > BC and AB > CA:
        top = centers[2]
        median_1 = centers[0]
        median_2 = centers[1]
    elif CA > AB and CA > BC:
        top = centers[1]
        median_1 = centers[0]
        median_2 = centers[2]
    elif BC > AB and BC > CA:
        top = centers[0]
        median_1 = centers[1]
        median_2 = centers[2]

    for point in central_points:
        if point not in (median_1, median_2):
            top = point

    try:
        perp_dist = calculate_perpendicular_distance(median_1, median_2, top)
        slope = calculate_slope(median_1, median_2)
    except ZeroDivisionError:
        return img, None

    if slope < 0 and perp_dist < 0:
        orientation = QR_OR_NORTH
        right = median_2
        bottom = median_1
    elif slope < 0 and perp_dist > 0:
        orientation = QR_OR_SOUTH
        right = median_1
        bottom = median_2
    elif slope > 0 and perp_dist < 0:
        orientation = QR_OR_EAST
        right = median_1
        bottom = median_2
    elif slope > 0 and perp_dist > 0:
        orientation = QR_OR_WEST
        right = median_2
        bottom = median_1


    qr_center_point = calculate_center(right, bottom)


    T = list() # top polygon
    R = list() # right polygon
    B = list() # bottom polygon

    try:
        for i in range(0, 4):
            T.append(central_points[top][i][0])
            R.append(central_points[right][i][0])
            B.append(central_points[bottom][i][0])
    except KeyError:
        return img, None

    try:
        extreme_T = find_extreme(qr_center_point, T)
        extreme_R = find_extreme(qr_center_point, R)
        extreme_B = find_extreme(qr_center_point, B)

        min_T = find_minimum(qr_center_point, T)
        min_R = find_minimum(qr_center_point, R)
        min_B = find_minimum(qr_center_point, B)
    except IndexError:
        return img, None

    point_alt_R = [tuple(point) for point in R if tuple(point) not in (extreme_R, min_R)]
    point_alt_B = [tuple(point) for point in B if tuple(point) not in (extreme_B, min_B)]

    eps = 2

    if orientation == QR_OR_WEST:
        second_point_R = list(list(filter(lambda e: e[0] == max(point[0] for point in point_alt_R), point_alt_R))[0])
        second_point_B = list(list(filter(lambda e: e[1] == min(point[1] for point in point_alt_B), point_alt_B))[0])
        second_point_R[1] -= eps
        second_point_B[0] += eps
        second_point_R = tuple(second_point_R)
        second_point_B = tuple(second_point_B)
    elif orientation == QR_OR_NORTH:
        second_point_R = list(list(filter(lambda e: e[1] == max(point[1] for point in point_alt_R), point_alt_R))[0])
        second_point_B = list(list(filter(lambda e: e[0] == max(point[0] for point in point_alt_B), point_alt_B))[0])
        second_point_R[0] += eps
        second_point_B[1] += eps
        second_point_R = tuple(second_point_R)
        second_point_B = tuple(second_point_B)
    elif orientation == QR_OR_EAST:
        second_point_R = list(list(filter(lambda e: e[1] == max(point[1] for point in point_alt_R), point_alt_R))[0])
        second_point_B = list(list(filter(lambda e: e[0] == min(point[0] for point in point_alt_B), point_alt_B))[0])
        second_point_R[1] += eps
        second_point_B[0] -= eps
        second_point_R = tuple(second_point_R)
        second_point_B = tuple(second_point_B)
    elif orientation == QR_OR_SOUTH:
        second_point_R = list(list(filter(lambda e: e[1] == min(point[1] for point in point_alt_R), point_alt_R))[0])
        second_point_B = list(list(filter(lambda e: e[0] == min(point[0] for point in point_alt_B), point_alt_B))[0])
        second_point_R[0] -= eps
        second_point_B[1] -= eps
        second_point_R = tuple(second_point_R)
        second_point_B = tuple(second_point_B)

    intersection = find_intersection(extreme_R, second_point_R, extreme_B, second_point_B)

    cv2.circle(img, intersection, 5, (0, 255, 0), 5)

    cv2.drawContours(img, [central_points[top]], 0, (0, 255, 0), 2)
    cv2.drawContours(img, [central_points[right]], 0, (255, 0, 0), 2)
    cv2.drawContours(img, [central_points[bottom]], 0, (0, 0, 255), 2)

    #print(orientation)
    #print(perp_dist)
    #print(slope)

    warped = geometrical_transformation(img, extreme_T, extreme_R, extreme_B, intersection)

    show_warped = False

    a1 = calculate_distance(top, right)
    a2 = calculate_distance(top, bottom)
    a3 = calculate_distance(right, intersection)
    a4 = calculate_distance(bottom, intersection)

    if abs(a1 - a2) < 20 and abs(a3 - a4) < 20:
        show_warped = True

    if show_warped:
        return img, warped
    else:
        return img, None

id_list = []
class_dict = {}

def register_qr(data):
    if data['id'] not in id_list:
        id_list.append(data['id'])
        product_class = data['class']
        try:
            temp = class_dict[product_class]
            temp.append(data)
            class_dict[product_class] = temp
        except KeyError:
            class_dict[product_class] = [data]


def report():
    print("QR_Code report:")
    for product_class in class_dict.keys():
        print(product_class, class_dict[product_class])

def main():

    #cap = cv2.VideoCapture("test_data/00003.MTS")
    cap = cv2.VideoCapture(0)

    try:
        while True:
            retval, frame = cap.read()

            img, warped = process_frame(frame)

            #if img is None:
                #continue

            cv2.imshow("Frame", img)
            if warped is not None:
                cv2.imshow("Warped", warped)

                decodedObjects = pyzbar.decode(warped)
                if not decodedObjects:
                    continue
                for obj in decodedObjects:
                    if obj.data:
                        print(obj.data)
                        byte_data = obj.data.decode('utf8').replace("'", '"')
                        json_data = json.loads(byte_data)
                        print(json_data)
                        register_qr(json_data)

            cv2.waitKey(1)

    except KeyboardInterrupt:
        report()

if __name__ == '__main__':
    main()