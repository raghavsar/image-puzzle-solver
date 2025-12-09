import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Tile import Tile
from Pixel import Pixel
from solver import simulate_solve_puzzle
from animation_generator import generate_puzzle_animation


font = cv2.FONT_HERSHEY_COMPLEX

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images = ["starry_night_rotate.png", "starry_night_translate.rgb",
              "starry_night_rotate.rgb",
              "starry_night_translate_irreg.png", "starry_night_rotate_irreg.png", "starry_night_translate_irreg.rgb",
              "starry_night_rotate_irreg.rgb",
              "mona_lisa_translate.png", "mona_lisa_rotate.png", "mona_lisa_translate.rgb", "mona_lisa_rotate.rgb",
              "mona_lisa_translate_irreg.png", "mona_lisa_rotate_irreg.png", "mona_lisa_translate_irreg.rgb",
              "mona_lisa_rotate_irreg.rgb",
              "sample1_translate.png", "sample1_rotate.png", "sample1_translate.rgb", "sample1_rotate.rgb",
              "sample1_translate_irreg.png", "sample1_rotate_irreg.png", "sample1_translate_irreg.rgb",
              "sample1_rotate_irreg.rgb",
              "sample2_translate.png", "sample2_rotate.png", "sample2_translate.rgb", "sample2_rotate.rgb",
              "sample2_translate_irreg.png", "sample2_rotate_irreg.png", "sample2_translate_irreg.rgb",
              "sample2_rotate_irreg.rgb",
              "sample3_translate.png", "sample3_rotate.png", "sample3_translate.rgb", "sample3_rotate.rgb",
              "sample3_translate_irreg.png", "sample3_rotate_irreg.png", "sample3_translate_irreg.rgb",
              "sample3_rotate_irreg.rgb"]
    # go through all images in the samples folder
    for image in images:
        print("Image: ", image)
        imagePath = "samples/" + image
        splitup = os.path.splitext(image)
        # get file extension
        fileExtension = splitup[1]
        # if .rgb file convert to a readable array
        if fileExtension == ".rgb":
            raw = np.fromfile(imagePath, dtype=np.uint8)
            if raw is None:
                 raise ValueError("Unable to read rgb file: " + imagePath)
            #convert rgb to bgr
            reshaped = raw.reshape(3, 800, 800,)
            transposed = np.transpose(reshaped, (1,2,0))
            img = cv2.cvtColor(transposed, cv2.COLOR_RGB2BGR)
        # else image must be .png so open it
        else:
            img = cv2.imread(imagePath)
            if img is None:
                raise ValueError("Unable to read png file: " + imagePath)

        # convert to grayscale to better find edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # declare lower threashold as black and upper as white
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # find all shapes & edges
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tiles = []
        numOfTiles = 0
        # loop through all edges for all shapes
        for i, contour in enumerate(contours):
            # approximate borders
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            sides = len(approx)

            if sides == 4:
                # Flatten contour points
                n = approx.ravel()
                x, y, w, h = cv2.boundingRect(approx)
                rect = cv2.minAreaRect(approx)
                (centerX, centerY), (width, height), angle = rect
                # print(angle)
                angle = int(round(angle))
                # print(angle)
                initial_position = (x, y)
                initial_rotation = angle

                '''
                tried to rotate another way with no success
                if angle != 90:
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    coords = sorted(box, key=lambda x: x[1])
                    if coords[2][0] < coords[0][0]:
                        angle += 90
                    if angle < 90:
                        rotation = -(90 - angle)
                    elif angle > 90:
                        rotation = (angle - 90)
                    rotationMatrix = cv2.getRotationMatrix2D((centerX,centerY), rotation, 1)
                    tile_img = cv2.warpAffine(img, rotationMatrix, (int(width), int(height)))
                else:
                    tile_img = img[y:y + h, x:x + w].copy()
                '''
                # upright angle should be either 0 or 90
                if angle != 90 and angle != 0:
                    box = cv2.boxPoints(rect)
                    box = np.float32(box)
                    coords = sorted(box, key=lambda x: x[1])
                    if coords[2][0] < coords[0][0]:
                        angle += 90
                    # if tilted right adjust to left
                    if 90 < angle and angle < 180 :
                        transform = np.float32([[0, height - 1],
                                      [0, 0],
                                      [width - 1, 0],
                                      [width - 1, height - 1]])
                    elif 0 < angle and angle < 90:
                        # else its tilted left so adjust to right
                        transform = np.float32([[0, 0],
                                    [width-1, 0],
                                    [width-1, height-1],
                                    [0, height-1]])
                    # rotate rectangle at box points
                    rotationMatrix = cv2.getPerspectiveTransform(box, transform)
                    # flags and border changes the way the border is
                    tile_img = cv2.warpPerspective(img, rotationMatrix, (int(width), int(height)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                    #cv2.imshow('rotated tile', tile_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    '''online says to use this for cordinates but doesnt seem to make much of a difference
                    pts = cv2.perspectiveTransform(approx.reshape(-1, 1, 2).astype(np.float32), rotationMatrix)
                    coordinates = [(int(pt[0][0]), int(pt[0][1])) for pt in pts]
                    '''
                    coordinates = [(pt[0] - x, pt[1] - y) for pt in approx.reshape(-1, 2)] # unsure if use this or above 2 lines
                else:
                    # Crop the tile image from the original image
                    tile_img = img[y:y+h, x:x+w].copy()
                    coordinates = [(pt[0] - x, pt[1] - y) for pt in approx.reshape(-1, 2)]
                # Adjust coordinates relative to cropped tile


                # Collect edge pixels relative to the tile
                pixels = []
                for p in contour:
                    px, py = p[0]
                    color = img[py, px]
                    pixels.append(Pixel(px - x, py - y, color[0], color[1], color[2]))

                # Draw borders and labels (optional)
                cv2.drawContours(img, [approx], 0, (255, 255, 255), 2)
                numOfTiles += 1
                cv2.putText(img, f"Tile {numOfTiles}", (n[0], n[1]+20), font, 0.4, (0, 255, 255))

                # Create Tile object
                tile = Tile(corners=coordinates, edges=pixels, image=tile_img, rotation=angle)
                tile.initial_rotation = angle
                tile.initial_position = (x, y)
                # tile.extracted_img = tile_img.copy()
                tiles.append(tile)


        print('Number of tiles found: ', numOfTiles)
        # cv2.imshow('image', img)
        # # press q to quite the picture and go through next one
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # =======Image Matching===========
        simulate_solve_puzzle(tiles, img)  # TODO: this function needs to be rewrite

        # # =======Animation Generation=======
        output_name = os.path.splitext(image)[0] + "_solution.gif"
        output_path = "outputs/"+output_name
        os.makedirs("outputs", exist_ok=True)
        print(output_name)
        generate_puzzle_animation(tiles, img, output_filename=output_path)
