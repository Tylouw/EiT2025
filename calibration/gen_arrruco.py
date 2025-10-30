import cv2 as cv

squaresX = 7
squaresY = 5
squareLength = 0.03   # 30 mm
markerLength = 0.022  # 22 mm
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)

# Use the constructor directly (works on OpenCV 4.9.x)
board = cv.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)

# Generate an A4-sized image (~300 DPI)
img = board.generateImage((3508, 2480))
cv.imwrite("charuco_board_A4.png", img)
print("Saved charuco_board_A4.png")
