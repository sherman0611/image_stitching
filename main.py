import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


# global variables
file_path_l, file_path_r = None, None
img_l, img_r = None, None
img_l_gray, img_r_gray = None, None

# sift parameters
contrastThreshold = 0.1


# main functions
def sift_detection(img, side):
    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold)
    kp = sift.detect(img, None)

    if side == "left":
        img_cp = img_l.copy()
    else:
        img_cp = img_r.copy()
        
    img_cp = cv2.drawKeypoints(img_cp, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    if side == "left":
        save_results(f'{file_path_l}_sift.png', img_cp)
        display_results(f'{file_path_l}_sift.png', side)
    else:
        save_results(f'{file_path_r}_sift.png', img_cp)
        display_results(f'{file_path_r}_sift.png', side)


def harris_detection(img, side):
    img = np.float32(img) # to 32 bit
    img_dest = cv2.cornerHarris(img, 2, 7, 0.04) # harris corner
    img_dest = cv2.dilate(img_dest, None) # mark corners

    if side == "left":
        img_cp = img_l.copy()
    else:
        img_cp = img_r.copy()
        
    # apply threshold to reduce number of detected features
    threshold = 0.01
    img_dest = img_dest > threshold * img_dest.max()
    img_cp[img_dest] = [0, 0, 255] # mark all detected corners in red

    if side == "left":    
        save_results(f'{file_path_l}_harris.png', img_cp)
        display_results(f'{file_path_l}_harris.png', side)
    else:    
        save_results(f'{file_path_r}_harris.png', img_cp)
        display_results(f'{file_path_r}_harris.png', side)


def sift_matching(img_l_gray, img_r_gray):
    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold)
    
    # find keypoints and descriptors
    kp_l, des_l = sift.detectAndCompute(img_l_gray, None)
    kp_r, des_r = sift.detectAndCompute(img_r_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
            matches = np.asarray(good)
        
    img = cv2.drawMatchesKnn(img_l, kp_l, img_r, kp_r, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    save_results(f'{file_path_l}_{file_path_r}_sift_matching.png', img)
    display_results(f'{file_path_l}_{file_path_r}_sift_matching.png', 'both')
    
    
def binary_matching(img_l_gray, img_r_gray):
    orb = cv2.ORB_create()
    
    # find keypoints and descriptors
    kp_l, des_l = orb.detectAndCompute(img_l_gray, None)
    kp_r, des_r = orb.detectAndCompute(img_r_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
     
    img = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    save_results(f'{file_path_l}_{file_path_r}_bin_matching.png', img)
    display_results(f'{file_path_l}_{file_path_r}_bin_matching.png', 'both')


def stitch(kp_l, kp_r, matches):

    src = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_l[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # find homography matrix
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
    # stitch images
    dst = cv2.warpPerspective(img_r, H, (img_r.shape[1] + img_l.shape[1], img_l.shape[0]))
    dst[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
    
    return dst


def sift_stitching(img_l_gray, img_r_gray):
    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold)
    
    kp_l, des_l = sift.detectAndCompute(img_l_gray, None)
    kp_r, des_r = sift.detectAndCompute(img_r_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
            matches = np.asarray(good)
    
    good = [g[0] for g in good]
    
    dst = stitch(kp_l, kp_r, good)
    
    save_results(f'{file_path_l}_{file_path_r}_sift_stitch.png', dst)
    display_results(f'{file_path_l}_{file_path_r}_sift_stitch.png', 'both')
    
    
def binary_stitching(img_l_gray, img_r_gray):
    orb = cv2.ORB_create()
    
    kp_l, des_l = orb.detectAndCompute(img_l_gray, None)
    kp_r, des_r = orb.detectAndCompute(img_r_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    
    dst = stitch(kp_l, kp_r, matches)

    save_results(f'{file_path_l}_{file_path_r}_binary_stitch.png', dst)
    display_results(f'{file_path_l}_{file_path_r}_binary_stitch.png', 'both')
    

# UI functions
def open_image(side):
    global file_path_l, file_path_r, img_l, img_r, img_l_gray, img_r_gray
    
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        if side == 'left':
            file_path_l = file_path
            img_l = cv2.imread(file_path_l)
            file_path_l = os.path.splitext(os.path.basename(file_path_l))[0] # get filename for saving results
            img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        elif side == 'right':
            file_path_r = file_path
            img_r = cv2.imread(file_path_r)
            file_path_r = os.path.splitext(os.path.basename(file_path_r))[0] # get filename for saving results
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        display_image(file_path, side)

        
def display_image(file_path, side):
    image = Image.open(file_path)
    
    original_width, original_height = image.size
    new_width = width // 2
    new_height = int((original_height / original_width) * new_width)
    image = image.resize((new_width, new_height))
    
    photo = ImageTk.PhotoImage(image)
    
    if side == 'left':
        left_image_label.config(image=photo)
        left_image_label.photo = photo
    elif side == 'right':
        right_image_label.config(image=photo)
        right_image_label.photo = photo
        
        
def reset():
    global file_path_l, file_path_r, img_l, img_r, img_l_gray, img_r_gray
    
    # clear display
    left_image_label.config(image='')
    right_image_label.config(image='')
    left_result_label.config(image='')
    right_result_label.config(image='')
    string_result_label.place_forget()
    
    file_path_l, file_path_r = None, None
    img_l, img_r = None, None
    img_l_gray, img_r_gray = None, None


def save_results(file_name, img):
    if not os.path.exists('results'):
        os.makedirs('results')
        
    cv2.imwrite(os.path.join('results', file_name), img)


def display_results(file_path, side):
    image = Image.open('results/' + file_path)
    
    if side != 'both':
        new_width = width // 2
    else:
        new_width = width
    img_width, img_height = image.size
    new_height = int((img_height / img_width) * new_width)
    image = image.resize((new_width, new_height))
    
    photo = ImageTk.PhotoImage(image)
    
    new_height = max(left_image_label.winfo_y() + left_image_label.winfo_height(),
                      right_image_label.winfo_y() + right_image_label.winfo_height()) + 10
    string_result_label.place(x=width//2, y=new_height, anchor='center')
    
    # string_result_label display fix
    if (string_result_label.winfo_height() == 1):
        new_height += string_result_label.winfo_height() + 10
    else:
        new_height += string_result_label.winfo_height() - 10
    
    if side == 'left':
        left_result_label.config(image=photo)
        left_result_label.photo = photo
        left_result_label.place(x=0, y=new_height)
    elif side == 'right':
        right_result_label.config(image=photo)
        right_result_label.photo = photo
        right_result_label.place(x=width // 2, y=new_height)
    elif side == 'both':
        right_result_label.place_forget() # remove right label
        left_result_label.config(image=photo)
        left_result_label.photo = photo
        left_result_label.place(x=0, y=new_height)


def confirm_selection():
    if img_l is None and img_r is None:
        messagebox.showinfo("Error", "Please select image first.")
        return
    
    selected_option = dropdown_var.get()
    
    if selected_option == "SIFT Feature Point Detection":
        if img_l is not None:
            sift_detection(img_l_gray, 'left')
        if img_r is not None:
            sift_detection(img_r_gray, 'right')
            
    elif selected_option == "Harris Corner Detection":
        if img_l is not None:
            harris_detection(img_l_gray, 'left')
        if img_r is not None:
            harris_detection(img_r_gray, 'right')
            
    elif img_l is None or img_r is None:
        messagebox.showinfo("Error", "Both images must be loaded.")
        return
            
    elif selected_option == "SIFT Feature Matching":
        sift_matching(img_l_gray, img_r_gray)
        
    elif selected_option == "Binary Feature Matching":
        binary_matching(img_l_gray, img_r_gray)
            
    elif selected_option == "SIFT Image Stitching":
        sift_stitching(img_l_gray, img_r_gray)
        
    elif selected_option == "Binary Image Stitching":
        binary_stitching(img_l_gray, img_r_gray)


# Render UI
window = tk.Tk()
window.title("Feature Detection & Image Stitch")

height = 850
width = 960
window.geometry(f'{width}x{height}')

# open file buttons
open_l_button = tk.Button(window, text="Select Left Image", command=lambda: open_image('left'))
open_l_button.place(x=(width // 4 - open_l_button.winfo_reqwidth() // 2), y=5)

open_r_button = tk.Button(window, text="Select Right Image", command=lambda: open_image('right'))
open_r_button.place(x=(3 * width // 4 - open_r_button.winfo_reqwidth() // 2), y=5)

# reset
reset_button = tk.Button(window, text="Reset Files", command=reset)
reset_button.place(x=(width // 2 - reset_button.winfo_reqwidth() // 2), y=5)

# display images labels
left_image_label = tk.Label(window)
left_image_label.place(x=0, y=open_l_button.winfo_reqheight()+10)

right_image_label = tk.Label(window)
right_image_label.place(x=width // 2, y=open_r_button.winfo_reqheight()+10)

# String results label
string_result_label = tk.Label(window, text="Results")

# display results labels
left_result_label = tk.Label(window)
right_result_label = tk.Label(window)

# Frame for bottom
bottom_frame = tk.Frame(window)
bottom_frame.place(relx=0.5, rely=1, anchor='s', y=-10)

# dropdown selection
options = ["SIFT Feature Point Detection", "Harris Corner Detection", "SIFT Feature Matching", "Binary Feature Matching",  "SIFT Image Stitching", "Binary Image Stitching"]
dropdown_var = tk.StringVar(window)
dropdown_var.set(options[0])
dropdown = tk.OptionMenu(bottom_frame, dropdown_var, *options)
dropdown.pack(side=tk.LEFT, padx=10)

# confirm button
confirm_button = tk.Button(bottom_frame, text="Confirm", command=confirm_selection)
confirm_button.pack(side=tk.LEFT, padx=10)

window.mainloop()
