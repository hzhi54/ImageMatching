import numpy as np
import cv2

def blue():
    img = cv2.imread("Hough.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 300, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/45, 150)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        # print(rho,theta)
        # print("[",rho,",",theta,"]")
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    rho = -560
    theta = 2.51
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    rho = 265
    theta = 2.51
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


    cv2.imwrite('blue.jpg', img)

def red():
    img = cv2.imread("Hough.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 300, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/150, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        if (x1-x2)/(y1-y2) < 0.1:
            # print("[",rho,",",theta,"]")
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('red.jpg', img)


def circle():
    img = cv2.imread('Hough.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                              param1=10, param2=20,
                              minRadius=30, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # cv2.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 0), 3)

    cv2.imwrite('circle.jpg', img)



if __name__ == "__main__":
    blue()
    red()
    circle()