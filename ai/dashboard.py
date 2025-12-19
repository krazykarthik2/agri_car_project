import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import time
from depthmap import DepthEstimator

class VehicleDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("VEHICLE COMMAND CENTER v1.0")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#121212')

        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize Depth Estimator from depthmap.py
        self.depth_estimator = DepthEstimator()
        
        # Mock Data
        self.fuel_pct = 85
        self.water_liters = 210
        self.mode = "MANUAL"

        self.setup_ui()
        
        # Bind Escape to exit
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        self.update_dashboard()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def setup_ui(self):
        # --- LEFT SIDEBAR (Gauges) ---
        self.sidebar = tk.Frame(self.root, bg='#1a1a1a', width=250)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.create_gauge(self.sidebar, "FUEL", "85%", "#00ffcc")
        self.create_gauge(self.sidebar, "WATER", "210L / 250L", "#00bcff")
        self.time_label = self.create_gauge(self.sidebar, "TIME", time.strftime("%H:%M"), "#ffcc00")

        # --- CENTER (Depth Estimation Feed) ---
        self.feed_container = tk.Frame(self.root, bg='black', bd=2, relief="sunken")
        self.feed_container.pack(expand=True, fill="both", padx=20, pady=10)
        
        self.video_label = tk.Label(self.feed_container, text="Initializing Camera...", bg="black", fg="#555")
        self.video_label.pack(expand=True, fill="both")

        # --- BOTTOM MENU (Modes) ---
        self.menu_frame = tk.Frame(self.root, bg='#121212', height=150)
        self.menu_frame.pack(side="bottom", fill="x", pady=20)

        modes = [
            ("BROADCAST\nSPRAY", "#ff4444"), 
            ("TARGETED\nSPRAY", "#ffbb33"), 
            ("SELF\nDRIVE", "#00C851"), 
            ("AUTO\nNAV", "#33b5e5")
        ]

        for text, color in modes:
            btn = tk.Button(self.menu_frame, text=text, bg='#1a1a1a', fg=color, 
                           font=("Orbitron", 10, "bold"), width=15, height=4,
                           relief="flat", activebackground=color)
            btn.pack(side="left", padx=20, expand=True)

    def create_gauge(self, parent, label, value, color):
        frame = tk.Frame(parent, bg='#1a1a1a', pady=20)
        frame.pack()
        tk.Label(frame, text=label, bg='#1a1a1a', fg="#888", font=("Arial", 10)).pack()
        lbl = tk.Label(frame, text=value, bg='#1a1a1a', fg=color, font=("Arial", 18, "bold"))
        lbl.pack()
        return lbl

    def update_dashboard(self):
        # Update Time
        if hasattr(self, 'time_label'):
             self.time_label.config(text=time.strftime("%H:%M"))
             
        # Capture Frame
        ret, frame = self.cap.read()
        if ret:
            # 1. Estimate Depth
            depth_map_color = self.depth_estimator.estimate_depth(frame)
            
            # 2. Blend
            combined = cv2.addWeighted(frame, 0.6, depth_map_color, 0.4, 0)
            
            # 3. Resize to fit container (Basic fixed size for perf, or dynamic)
            # A fixed reasonable size is good for FPS
            combined = cv2.resize(combined, (800, 450))
            
            # 4. Convert for Tkinter
            img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # 5. Update Label
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk # Keep reference!
            
        self.root.after(10, self.update_dashboard)

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDashboard(root)
    root.mainloop()