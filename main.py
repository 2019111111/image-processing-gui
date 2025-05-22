import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing GUI")

        self.original_image = None
        self.result_image = None

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.original_image_label = tk.Label(self.image_frame, text="Original")
        self.original_image_label.grid(row=0, column=0)
        self.processed_image_label = tk.Label(self.image_frame, text="Processed")
        self.processed_image_label.grid(row=0, column=1)

        self.original_img_display = tk.Label(self.image_frame)
        self.original_img_display.grid(row=1, column=0, padx=10, pady=10)
        self.processed_img_display = tk.Label(self.image_frame)
        self.processed_img_display.grid(row=1, column=1, padx=10, pady=10)

        tk.Button(root, text="Upload Image", command=self.load_image).pack(pady=5)

        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(padx=10, pady=10, fill="x")

        self.filter_tree = ttk.Treeview(self.controls_frame, height=10)
        self.filter_tree.heading("#0", text="Filter Categories")
        self.filter_tree.pack(side="left", fill="y")

        self.filter_structure = {
            "Noise Filters": ["Salt & Pepper", "Gaussian Noise", "Poisson Noise"],
            "Point Transformations": ["Brightness/Contrast", "Histogram", "Histogram Equalization"],
            "Local Filters": ["Low Pass", "High Pass", "Median Filter", "Averaging Filter"],
            "Edge Detection": ["Laplacian", "Sobel H", "Sobel V", "Prewitt H", "Prewitt V", "LoG", "Canny"],
            "Hough Transform": ["Hough Lines", "Hough Circles"],
            "Morphology": ["Dilation", "Erosion", "Opening", "Closing"]
        }

        for category, filters in self.filter_structure.items():
            cat_id = self.filter_tree.insert("", "end", text=category)
            for f in filters:
                self.filter_tree.insert(cat_id, "end", text=f)

        self.filter_tree.bind("<Double-1>", self.on_filter_select)

        self.sliders_frame = tk.Frame(self.controls_frame)
        self.sliders_frame.pack(side="right", fill="both", expand=True)

        self.brightness_scale = tk.Scale(self.sliders_frame, from_=-100, to=100, orient="horizontal", label="Brightness")
        self.brightness_scale.pack(fill="x")
        self.contrast_scale = tk.Scale(self.sliders_frame, from_=1, to=300, orient="horizontal", label="Contrast (%)")
        self.contrast_scale.set(100)
        self.contrast_scale.pack(fill="x")
        self.kernel_scale = tk.Scale(self.sliders_frame, from_=1, to=31, orient="horizontal", label="Kernel Size (odd only)")
        self.kernel_scale.set(5)
        self.kernel_scale.pack(fill="x")

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_image = Image.open(path).convert("RGB")
            self.result_image = self.original_image.copy()
            self.show_images()

    def show_images(self):
        original = self.original_image.resize((300, 240))
        processed = self.result_image.resize((300, 240))
        orig_tk = ImageTk.PhotoImage(original)
        proc_tk = ImageTk.PhotoImage(processed)
        self.original_img_display.config(image=orig_tk)
        self.original_img_display.image = orig_tk
        self.processed_img_display.config(image=proc_tk)
        self.processed_img_display.image = proc_tk

    def on_filter_select(self, event):
        selected_item = self.filter_tree.selection()[0]
        filter_name = self.filter_tree.item(selected_item, "text")
        self.apply_filter(filter_name)

    def apply_filter(self, filter_name):
        if self.original_image is None:
            return

        img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        kernel_size = self.kernel_scale.get()
        if kernel_size % 2 == 0:
            kernel_size += 1

        result = img_cv.copy()

        try:
            if filter_name == "Salt & Pepper":
                noisy = np.array(self.original_image)
                prob = 0.05
                thres = 1 - prob
                rnd = np.random.rand(*noisy.shape[:2])
                noisy[rnd < prob] = 0
                noisy[rnd > thres] = 255
                result = noisy

            elif filter_name == "Gaussian Noise":
                row, col, ch = img_cv.shape
                gauss = np.random.normal(0, 25, (row, col, ch)).astype('uint8')
                result = cv2.add(img_cv, gauss)

            elif filter_name == "Poisson Noise":
                vals = len(np.unique(img_cv))
                vals = 2 ** np.ceil(np.log2(vals))
                noisy = np.random.poisson(img_cv * vals) / float(vals)
                result = np.clip(noisy, 0, 255).astype('uint8')

            elif filter_name == "Brightness/Contrast":
                brightness = self.brightness_scale.get()
                contrast = self.contrast_scale.get() / 100.0

                result = img_cv.astype(np.float32)
                result = result * contrast + brightness
                result = np.clip(result, 0, 255).astype(np.uint8)

            elif filter_name == "Histogram":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                plt.hist(gray.ravel(), 256, [0, 256])
                plt.title('Histogram')
                plt.show()
                return

            elif filter_name == "Histogram Equalization":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                result = cv2.equalizeHist(gray)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            elif filter_name == "Low Pass":
                result = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)

            elif filter_name == "High Pass":
                blur = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
                result = cv2.subtract(img_cv, blur)

            elif filter_name == "Median Filter":
                result = cv2.medianBlur(img_cv, kernel_size)

            elif filter_name == "Averaging Filter":
                result = cv2.blur(img_cv, (kernel_size, kernel_size))

            elif filter_name == "Laplacian":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                result = cv2.Laplacian(gray, cv2.CV_64F)
                result = cv2.convertScaleAbs(result)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            elif filter_name == "Sobel H":
                result = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=kernel_size)
                result = cv2.convertScaleAbs(result)

            elif filter_name == "Sobel V":
                result = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=kernel_size)
                result = cv2.convertScaleAbs(result)

            elif filter_name == "Prewitt H":
                kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                result = cv2.filter2D(img_cv, -1, kernel)

            elif filter_name == "Prewitt V":
                kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                result = cv2.filter2D(img_cv, -1, kernel)

            elif filter_name == "LoG":
                blur = cv2.GaussianBlur(img_cv, (3,3), 0)
                result = cv2.Laplacian(blur, cv2.CV_64F)
                result = cv2.convertScaleAbs(result)

            elif filter_name == "Canny":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                result = cv2.Canny(gray, 100, 200)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            elif filter_name == "Hough Lines":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)

            elif filter_name == "Hough Circles":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0,:]:
                        cv2.circle(result, (i[0], i[1]), i[2], (0,255,0), 2)

            elif filter_name == "Dilation":
                result = cv2.dilate(img_cv, np.ones((5,5),np.uint8), iterations=1)

            elif filter_name == "Erosion":
                result = cv2.erode(img_cv, np.ones((5,5),np.uint8), iterations=1)

            elif filter_name == "Opening":
                result = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

            elif filter_name == "Closing":
                result = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        except Exception as e:
            print(f"Error applying filter {filter_name}: {e}")
            result = img_cv

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.result_image = Image.fromarray(result_rgb)
        self.show_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()
