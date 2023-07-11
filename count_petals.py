import os
from PIL import Image
from plantcv import plantcv as pcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import unionfind as UF
import utils
import csv
import pandas as pd

def verify_image(filepath):
    try:
        image = Image.open(filepath)
        image.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {e}")
        return False

def count_petals(filepath,visualize=False):
    # load image
    img = cv2.imread(filepath) # read in the image
    orig_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # convert it from BGR to RGB (used for visualization)

    # resize and blur image
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    blurred_img = cv2.medianBlur(resized_img, 15)

    # use the plantcv naive bayes classifier to find the flower pixels in the tiny blurred image
    mask = pcv.naive_bayes_classifier(rgb_img=blurred_img,pdf_file="pdfs.txt")

    # Find the largest contiguous region in the foreground mask -- that's our biggest flower
    ret, thresh = cv2.threshold(np.array(mask['fg']),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # find connected components
    labels = utils.get_largest_element(thresh) # find the one that's biggest
    labels = ndimage.binary_fill_holes(labels) # fill in any holes in the biggest flower

    # find a padded bounding box that contains that flower
    y_vals, x_vals = np.nonzero(labels)
    pad = 10
    y_min = np.min(y_vals) - pad
    y_max = np.max(y_vals) + pad
    x_min = np.min(x_vals) - pad
    x_max = np.max(x_vals) + pad

    # extract a crop of just the biggest flower for visualizations
    cropped_img = resized_img[y_min:y_max, x_min:x_max, :]

    # get just the foreground/background labels in the cropped region
    labels = labels[y_min:y_max, x_min:x_max]
    com = np.asarray(ndimage.center_of_mass(labels))

    # compute the shortest distance from every pixel in the foreground to the background
    edt = np.ones_like(labels)
    edt[int(com[0]), int(com[1])] = False
    edt = ndimage.distance_transform_edt(edt)
    edt[labels == 0] = 0

    # compute the persistence of that distance transform
    pers = sorted(UF.persistence(edt),reverse=True)
    thr = 8
    counter = 1
    while pers[counter][0] > thr:
        #print(counter, pers[counter][0], sep='\t')
        counter += 1
    persf = pers[:counter]
    births = np.zeros((len(persf), 2), dtype=int)
    for i in range(len(persf)):
        births[i] = persf[i][2]
    births = births.T

    deaths = np.zeros((len(persf)-1, 2), dtype=int)
    for i in range(1,len(persf)):
        deaths[i-1] = persf[i][1]
    deaths = deaths.T

    num_petals = deaths.shape[1]+1

    if visualize:
        fig, axes = plt.subplots(1,4,figsize=(15,4.5))
        axes = np.atleast_1d(axes).ravel()
        axes[0].imshow(orig_rgb);
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB));
        axes[2].imshow(labels);
        axes[3].imshow(edt)

        for i in range(1,len(axes)):
            axes[i].axis('off')
            axes[i].scatter([com[1]], [com[0]], color='k')
            axes[i].scatter(births[1], births[0], color='r', marker='^', s=75)
            axes[i].scatter(deaths[1], deaths[0], color='m', marker='v', s=75)

        fig.suptitle(num_petals, fontsize=20)
        fig.tight_layout();

        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        visualization_file = os.path.join('visualizations',os.path.basename(filepath))
        plt.savefig(visualization_file, dpi=100, bbox_inches='tight')

    return num_petals

def write_csv(filepath, output_file, num_petals):
    if os.path.isfile(output_file):
        updated = False
        rows = []

        with open(output_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == filepath:
                    row[1] = num_petals
                    updated = True
                rows.append(row)

        if not updated:
            rows.append([filepath, num_petals])

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
    else:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file", "auto_count"])
            writer.writerow([filepath, num_petals])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python count_petals.py <filepath> [output_file] [visualize] [compare_file]")
    else:
        filepath = sys.argv[1]
        output_file = './petal_counts.csv'
        compare_file = None
        visualize = True

        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            if not '.csv' in output_file:
                print('output_file must be a csv')
                exit()

        if len(sys.argv) >= 4:
            visualize = sys.argv[3].lower() == "true"

        if len(sys.argv) == 5:
            compare_file = sys.argv[4]
            if not '.csv' in compare_file:
                print('compare_file must be a csv')
                exit()

        if os.path.isfile(filepath):
            num_petals = count_petals(filepath, visualize)
            if num_petals is not None:
                write_csv(filepath, output_file, num_petals)
            else:
                print("Unable to count petals.")

        elif os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        num_petals = count_petals(os.path.join(root, file), visualize)
                        if num_petals is not None:
                            write_csv(file, output_file, num_petals)
                        else:
                            print("Unable to count petals.")

            if compare_file is not None:
                # Read the CSV files
                df1 = pd.read_csv(output_file)
                df2 = pd.read_csv(compare_file)

                # Merge the dataframes based on the shared column name
                # Merge is creating duplicate rows, why?
                merged_df = pd.merge(df1, df2, on='file').drop_duplicates()

                # Create a scatter plot
                plt.scatter(merged_df['manual_count'], merged_df['auto_count'])
                plt.xlabel('manual')
                plt.ylabel('automated')
                plt.title('Petal Counts')

                # Save the scatter plot to a file
                if not os.path.exists('visualizations'):
                    os.makedirs('visualizations')

                visualization_file = os.path.join('visualizations',os.path.basename('scatter_plot.png'))
                plt.savefig(visualization_file)




        else:
            print("Invalid file path.")