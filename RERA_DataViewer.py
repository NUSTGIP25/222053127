
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.filedialog import askdirectory
from tkintermapview import TkinterMapView
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.warp import transform_bounds
from PIL import Image, ImageTk
import shutil
import webbrowser
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import math

os.environ["PROJ_DATA"] = r"C:\Users\USER\Documents\NUST\2025\GIP\Project\rera-python\myenv\Lib\site-packages\pyproj\proj_dir\share\proj"
os.environ["GDAL_DATA"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "share", "gdal") 
os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "share", "proj")


class MapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RERA Version 2.0 (Data Viewer)")
        self.geometry("1200x700")
        self.configure(bg="#f0f0f0")
        
        # Create a style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4CAF50", foreground="black", borderwidth=1)
        self.style.configure("Green.TButton", background="#4CAF50", foreground="white")
        self.style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#f0f0f0")
        self.style.configure("TCheckbutton", background="#f0f0f0")

        # Layout
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        
        self.sidebar = ttk.Frame(self.main_frame, width=300)
        self.sidebar.pack(side="left", fill="y")
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.sidebar)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs
        self.layers_tab = ttk.Frame(self.notebook)

        self.wms_tab = ttk.Frame(self.notebook)

        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.layers_tab, text="Layers")

        self.notebook.add(self.wms_tab, text="WMS")

        self.notebook.add(self.settings_tab, text="Settings")

        # Map widget
        self.map_widget = TkinterMapView(self.main_frame, width=900, height=700, corner_radius=0)
        self.map_widget.pack(side="right", fill="both", expand=True)
        self.map_widget.set_position(-22.5597, 17.0832)  # Windhoek
        self.map_widget.set_zoom(6)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

        # Setup variables
        self.layer_vars = {}
        self.layer_objects = {}
        self.data_folder = "data"
        self.wms_servers = []
        
        # Load settings
        self.settings = {
            "raster_opacity": 1.0,
            "vector_color": "#3388ff",
            "vector_width": 2,
            "max_display_features": 20000
        }
        
        # Populate the UI
        self.populate_layers_tab()

        self.populate_wms_tab()
        
        self.populate_settings_tab()

    def populate_layers_tab(self):
        ttk.Label(self.layers_tab, text="Available Datasets", style="Header.TLabel").pack(pady=10)
        
        # Create a frame for the layer filters
        filter_frame = ttk.Frame(self.layers_tab)
        filter_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side="left", padx=2)
        
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        filter_entry.pack(side="left", fill="x", expand=True, padx=2)
        self.filter_var.trace("w", lambda *args: self.filter_layers())
        
        # Create a canvas with scrollbar for layers
        self.canvas_frame = ttk.Frame(self.layers_tab)
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame)
        scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Load data layers
        self.load_available_layers()
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(self.layers_tab)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        refresh_btn = ttk.Button(button_frame, text="🔄 Refresh", command=self.reload_layers)
        refresh_btn.pack(side="left", padx=5)
        
        add_btn = ttk.Button(button_frame, text="➕ Add Data", command=self.add_data)
        add_btn.pack(side="right", padx=5)

    def filter_layers(self):
        filter_text = self.filter_var.get().lower()
        
        # Hide all frames first
        for widget in self.scrollable_frame.winfo_children():
            widget.pack_forget()
        
        # Show only filtered frames
        for fname, frame in self.layer_frames.items():
            if filter_text in fname.lower():
                frame.pack(fill="x", padx=2, pady=2)

    def load_available_layers(self):
        # Clear existing layers
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.layer_frames = {}
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        # Group files by type
        vector_files = []
        raster_files = []
        
        for fname in sorted(os.listdir(self.data_folder)):
            path = os.path.join(self.data_folder, fname)
            if fname.endswith((".shp", ".geojson")):
                vector_files.append(fname)
            elif fname.endswith(".tif"):
                raster_files.append(fname)
        
        # Add vector files
        if vector_files:
            ttk.Label(self.scrollable_frame, text="Vector Layers", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=5, pady=5)
            for fname in vector_files:
                self.add_layer_widget(fname)
        
        # Add raster files
        if raster_files:
            ttk.Label(self.scrollable_frame, text="Raster Layers", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=5, pady=5)
            for fname in raster_files:
                self.add_layer_widget(fname)
                
        if not vector_files and not raster_files:
            ttk.Label(self.scrollable_frame, text="No data found. Add data using the '+ Add Data' button").pack(pady=20)

    def add_layer_widget(self, fname):
        path = os.path.join(self.data_folder, fname)
        var = tk.BooleanVar()
        
        frame = ttk.Frame(self.scrollable_frame)
        self.layer_frames[fname] = frame
        frame.pack(fill="x", padx=2, pady=2)
        
        is_raster = fname.endswith(".tif")
        icon = "🟦" if is_raster else "📍"
        
        chk = ttk.Checkbutton(frame, text=f"{icon} {fname}", variable=var,
                             command=lambda f=fname, v=var: self.toggle_layer(f, v))
        chk.pack(side="left", fill="x", expand=True)
        
        # Button frame for actions
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="right")
        
        # Info button
        info_btn = ttk.Button(btn_frame, text="ℹ️", width=2,
                             command=lambda p=path: self.show_layer_info(p))
        info_btn.pack(side="left", padx=1)
        
        # Download button
        dl_btn = ttk.Button(btn_frame, text="⬇️", width=2,
                           command=lambda p=path: self.download_file(p))
        dl_btn.pack(side="left", padx=1)
        
        self.layer_vars[fname] = var

    def reload_layers(self):
        # Unload all layers
        for filename in list(self.layer_objects.keys()):
            self.unload_layer(filename)
            
        # Reset checkboxes
        for var in self.layer_vars.values():
            var.set(False)
            
        # Reload available layers
        self.load_available_layers()
        self.status_var.set("Layers refreshed")

    def add_data(self):
        messagebox.showinfo("Add Data", "Please place your data files in the 'data' folder and click 'Refresh'.")

    def populate_wms_tab(self):
        ttk.Label(self.wms_tab, text="WMS Services", style="Header.TLabel").pack(pady=10)
        
        # WMS Servers frame
        wms_frame = ttk.Frame(self.wms_tab)
        wms_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Launch WMS viewer button
        launch_btn = ttk.Button(self.wms_tab, text="🌍 Launch Web Viewer", 
                               command=self.launch_wms_viewer, style="Green.TButton")
        launch_btn.pack(pady=10, padx=10, fill="x")
        
        # WMS Configuration button
        config_btn = ttk.Button(self.wms_tab, text="⚙️ Configure WMS Map", 
                               command=self.configure_wms)
        config_btn.pack(pady=5, padx=10, fill="x")

    def configure_wms(self):
        # This would open a dialog to edit the WMS configuration
        messagebox.showinfo("WMS Configuration", 
                          "This feature would allow you to configure WMS layers and endpoints.\n\n"
                          "Currently, the web viewer uses pre-configured WMS layers.")

    def populate_settings_tab(self):
        ttk.Label(self.settings_tab, text="App Settings", style="Header.TLabel").pack(pady=10)
        
        # Settings frame
        settings_frame = ttk.Frame(self.settings_tab)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Raster opacity
        ttk.Label(settings_frame, text="Raster Opacity:").pack(anchor="w", pady=5)
        
        opacity_frame = ttk.Frame(settings_frame)
        opacity_frame.pack(fill="x", pady=5)
        
        self.opacity_var = tk.DoubleVar(value=self.settings["raster_opacity"])
        opacity_scale = ttk.Scale(opacity_frame, from_=0.1, to=1.0, 
                                variable=self.opacity_var, orient="horizontal")
        opacity_scale.pack(side="left", fill="x", expand=True)
        
        opacity_label = ttk.Label(opacity_frame, width=4, 
                                 text=f"{int(self.opacity_var.get()*100)}%")
        opacity_label.pack(side="right", padx=5)
        
        self.opacity_var.trace("w", lambda *args: 
                             opacity_label.config(text=f"{int(self.opacity_var.get()*100)}%"))
        
        # Max features to display
        ttk.Label(settings_frame, text="Max Features to Display:").pack(anchor="w", pady=5)
        
        max_features_frame = ttk.Frame(settings_frame)
        max_features_frame.pack(fill="x", pady=5)
        
        self.max_features_var = tk.IntVar(value=self.settings["max_display_features"])
        max_features_entry = ttk.Entry(max_features_frame, textvariable=self.max_features_var, width=8)
        max_features_entry.pack(side="left")
        
        # Apply button
        apply_btn = ttk.Button(settings_frame, text="Apply Settings", 
                              command=self.apply_settings, style="Green.TButton")
        apply_btn.pack(pady=15)

    def apply_settings(self):
        # Save settings
        self.settings["raster_opacity"] = self.opacity_var.get()
        self.settings["max_display_features"] = self.max_features_var.get()
        
        # Apply to existing layers
        self.status_var.set("Settings applied")
        
        # Reload active layers to apply new settings
        active_layers = [fname for fname, var in self.layer_vars.items() if var.get()]
        
        for filename in active_layers:
            self.unload_layer(filename)
            self.layer_vars[filename].set(True)
            self.toggle_layer(filename, self.layer_vars[filename])

    def show_layer_info(self, path):
        filename = os.path.basename(path)
        info_text = f"File: {filename}\nPath: {path}\n\n"
        
        try:
            if filename.endswith(".tif"):
                with rasterio.open(path) as src:
                    info_text += f"Type: Raster (GeoTIFF)\n"
                    info_text += f"Size: {src.width} x {src.height} pixels\n"
                    info_text += f"Bands: {src.count}\n"
                    info_text += f"CRS: {src.crs}\n"
                    bounds = src.bounds
                    info_text += f"Bounds: {bounds}\n"
                    info_text += f"Resolution: {src.res}\n"
                    
                    # Sample data stats from first band
                    data = src.read(1, masked=True)
                    info_text += f"Data Min: {data.min():.4f}\n"
                    info_text += f"Data Max: {data.max():.4f}\n"
                    info_text += f"Data Mean: {data.mean():.4f}\n"
            else:
                gdf = gpd.read_file(path)
                info_text += f"Type: Vector ({filename.split('.')[-1]})\n"
                info_text += f"Features: {len(gdf)}\n"
                info_text += f"CRS: {gdf.crs}\n"
                info_text += f"Geometry Types: {set(gdf.geometry.geom_type)}\n"
                info_text += f"Attributes: {', '.join(gdf.columns)}\n"
                bounds = gdf.total_bounds
                info_text += f"Bounds: {bounds}\n"
                    
        except Exception as e:
            info_text += f"\nError reading file: {str(e)}"
            
        # Create a dialog with the info
        info_window = tk.Toplevel(self)
        info_window.title(f"Layer Info: {filename}")
        info_window.geometry("500x400")
        
        info_text_widget = tk.Text(info_window, wrap="word")
        info_text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        info_text_widget.insert("1.0", info_text)
        info_text_widget.config(state="disabled")
        
        close_btn = ttk.Button(info_window, text="Close", command=info_window.destroy)
        close_btn.pack(pady=10)

    def launch_wms_viewer(self):
        map_path = os.path.abspath("map.html")
        if os.path.exists(map_path):
            webbrowser.open(f"file://{map_path}")
            self.status_var.set("Web viewer launched in browser")
        else:
            messagebox.showerror("Not Found", "map.html was not found in the application folder.")

    def download_file(self, filepath):
        dest_dir = askdirectory(title="Select Download Folder")
        if dest_dir:
            try:
                filename = os.path.basename(filepath)
                shutil.copy(filepath, os.path.join(dest_dir, filename))
                self.status_var.set(f"{filename} downloaded successfully")
                messagebox.showinfo("Success", f"{filename} downloaded successfully to {dest_dir}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def toggle_layer(self, filename, var):
        path = os.path.join(self.data_folder, filename)
        
        if var.get():
            self.status_var.set(f"Loading {filename}...")
            self.update_idletasks()  # Update UI
            
            if filename.endswith(".tif"):
                self.load_raster_layer(path, filename)
            else:
                self.load_vector_layer(path, filename)
                
            self.status_var.set(f"{filename} loaded")
        else:
            self.unload_layer(filename)
            self.status_var.set(f"{filename} removed")

    def get_dataset_style(self, filename):
        styles = {
            # Vector styles - format: [color, width, opacity]
            "Land Tenure.geojson": ["#8BC34A", 2, 0.6],  # Light green for land use
            "Major Rivers.geojson": ["#2196F3", 3, 0.8],  # Blue for rivers
            "NamRoads.geojson": ["#FF5722", 2, 1.0],      # Orange-red for roads
            "Perennial Catchment Areas.geojson": ["#4FC3F7", 2, 0.5],  # Light blue for catchment
            "Powerlines.geojson": ["#F44336", 3, 1.0],    # Red for powerlines
            "R.E. Sources.geojson": ["#4CAF50", 2, 1.0],  # Green for renewable sources
            "Substations.geojson": ["#FFC107", 2, 1.0],   # Amber for substations
            
            # Raster styles - format: [colormap, min_val, max_val]
            "DNI.tif": ["viridis", None, None],           # viridis colormap for solar irradiance
            "NAM_wind-speed_100m.tif": ["plasma", 0, 12], # plasma colormap for wind speed
            "NamDEM.tif": ["terrain", None, None],        # terrain colormap for elevation
        }
        
        return styles.get(filename, None)

    def unload_layer(self, filename):
        for obj in self.layer_objects.get(filename, []):
            try:
                obj.delete()
            except:
                pass
        self.layer_objects[filename] = []

    def load_vector_layer(self, path, filename):
        try:
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf[gdf.geometry.is_valid & gdf.geometry.notnull()]

            # Apply simplification if there are many features
            max_features = self.settings["max_display_features"]
            if len(gdf) > max_features:
                self.status_var.set(f"Simplifying {len(gdf)} features for display...")
                self.update_idletasks()
                
                # Sample features if there are too many
                if len(gdf) > max_features * 2:
                    gdf = gdf.sample(max_features)
                
                # Apply simplification
                gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0005, preserve_topology=True)

            obj_refs = []
            
            # Get dataset-specific styling if available
            dataset_style = self.get_dataset_style(filename)
            if dataset_style:
                color, width, _ = dataset_style  # opacity not used
            else:
                color = self.settings["vector_color"]
                width = self.settings["vector_width"]

            for _, row in gdf.iterrows():
                geom = row.geometry
                try:
                    # Handle point geometries with styling
                    if geom.geom_type == "Point":
                        lat, lon = geom.y, geom.x
                        
                        # Create a tooltip from attributes if available
                        tooltip = None
                        if 'name' in row:
                            tooltip = str(row['name'])
                        elif 'label' in row:
                            tooltip = str(row['label'])
                        elif 'NAME' in row:
                            tooltip = str(row['NAME'])
                        elif 'id' in row:
                            tooltip = str(row['id'])
                        
                        # For point features, we'll create small colored circles using polygons
                        # since TkinterMapView doesn't support custom marker colors
                        
                        # Create a small circle around the point
                        circle_radius = 0.09  # Adjust this value to make points bigger/smaller
                        
                        # Generate circle coordinates
                        import math
                        num_points = 12  # Number of points to create the circle
                        circle_coords = []
                        
                        for i in range(num_points):
                            angle = 2 * math.pi * i / num_points
                            circle_lat = lat + circle_radius * math.cos(angle)
                            circle_lon = lon + circle_radius * math.sin(angle)
                            circle_coords.append((circle_lat, circle_lon))
                        
                        # Close the circle
                        circle_coords.append(circle_coords[0])
                        
                        # Create the circular polygon with dataset-specific color
                        circle_obj = self.map_widget.set_polygon(
                            circle_coords,
                            fill_color=color,
                            outline_color=color,
                            border_width=2
                        )
                        obj_refs.append(circle_obj)
                        
                        # Add a text label if we have tooltip information
                        if tooltip and len(str(tooltip)) < 20:  # Only show short labels
                            text_obj = self.map_widget.set_text(
                                lat + circle_radius * 1.5,  # Position text slightly above the circle
                                lon,
                                text=str(tooltip)[:15],  # Limit text length
                                font=("Arial", 8)
                            )
                            obj_refs.append(text_obj)

                    # Handle line geometries
                    elif geom.geom_type in ["LineString", "MultiLineString"]:
                        if geom.geom_type == "MultiLineString":
                            parts = geom.geoms
                        else:
                            parts = [geom]
                            
                        for part in parts:
                            coords = list(part.coords)
                            path_obj = self.map_widget.set_path(
                                [(lat, lon) for lon, lat in coords],
                                color=color,
                                width=width
                            )
                            obj_refs.append(path_obj)
                            
                    # Handle polygon geometries
                    elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                        if geom.geom_type == "MultiPolygon":
                            parts = geom.geoms
                        else:
                            parts = [geom]
                            
                        for part in parts:
                            # Get exterior coordinates
                            exterior_coords = list(part.exterior.coords)
                            
                            # Create polygon path
                            polygon_obj = self.map_widget.set_polygon(
                                [(lat, lon) for lon, lat in exterior_coords],
                                fill_color=color,
                                outline_color=color,
                                border_width=width
                            )
                            obj_refs.append(polygon_obj)

                except Exception as ge:
                    print(f"Skipping bad geometry in {filename}: {ge}")

            self.layer_objects[filename] = obj_refs

        except Exception as e:
            self.status_var.set(f"Error loading {filename}")
            messagebox.showerror("Load Error", f"Failed to load {filename}\n{str(e)}")

    def load_raster_layer(self, path, filename):
            try:
                with rasterio.open(path) as src:
                    # Get the bounds in the original CRS
                    left, bottom, right, top = src.bounds
                    
                    # Transform bounds to EPSG:4326 (lat/lon) if needed
                    if src.crs != "EPSG:4326":
                        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", 
                                                                left, bottom, right, top)
                    
                    # Read the raster data
                    data = src.read(1, masked=True)  # Read first band
                    
                    # Get dataset-specific styling
                    dataset_style = self.get_dataset_style(filename)
                    if dataset_style:
                        cmap_name, min_val, max_val = dataset_style
                        cmap = getattr(cm, cmap_name)
                        
                        # Use specified min/max if provided, otherwise use data min/max
                        vmin = min_val if min_val is not None else data.min()
                        vmax = max_val if max_val is not None else data.max()
                    else:
                        # Default colormap if no specific styling defined
                        cmap = cm.viridis
                        vmin = data.min()
                        vmax = data.max()
                    
                    # Set up normalization
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    
                    # Create colored tiles at appropriate resolution
                    h, w = data.shape
                    
                    # Determine appropriate step size based on raster dimensions
                    target_resolution = 50  # Target number of cells in each dimension
                    step_y = max(1, h // target_resolution)
                    step_x = max(1, w // target_resolution)
                    
                    self.status_var.set(f"Rendering raster with {w//step_x}x{h//step_y} cells...")
                    self.update_idletasks()
                    
                    obj_refs = []
                    
                    # Create polygons for each cell
                    for y in range(0, h, step_y):
                        for x in range(0, w, step_x):
                            # Get cell bounds in pixel coordinates
                            cell_left = x
                            cell_top = y
                            cell_right = min(x + step_x, w)
                            cell_bottom = min(y + step_y, h)
                            
                            # Transform to geographic coordinates
                            ul_x, ul_y = src.transform * (cell_left, cell_top)
                            lr_x, lr_y = src.transform * (cell_right, cell_bottom)
                            
                            # If not in EPSG:4326, transform points
                            if src.crs != "EPSG:4326":
                                ul_x, ul_y = transform_bounds(src.crs, "EPSG:4326", ul_x, ul_y, ul_x, ul_y)
                                lr_x, lr_y = transform_bounds(src.crs, "EPSG:4326", lr_x, lr_y, lr_x, lr_y)
                                
                            # Sample data value for cell (mean)
                            cell_data = data[cell_top:cell_bottom, cell_left:cell_right]
                            if cell_data.size > 0 and not np.all(cell_data.mask):
                                # Only create polygon for non-masked data
                                if hasattr(cell_data, 'mask'):
                                    cell_data = cell_data.filled(np.nan)  # Replace masked values with NaN
                                
                                # Calculate mean value (ignoring NaNs)
                                mean_val = np.nanmean(cell_data)
                                
                                if not np.isnan(mean_val):
                                    # Get color from colormap
                                    rgba = cmap(norm(mean_val))
                                    hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
                                    
                                    # Create polygon
                                    corners = [
                                        (ul_y, ul_x),  # Upper left (latitude, longitude)
                                        (ul_y, lr_x),  # Upper right
                                        (lr_y, lr_x),  # Lower right
                                        (lr_y, ul_x),  # Lower left
                                    ]
                                    
                                    polygon = self.map_widget.set_polygon(
                                        corners,
                                        fill_color=hex_color,
                                        outline_color="",  # No outline
                                        border_width=0
                                    )
                                    obj_refs.append(polygon)
                    
                    # Create descriptive legend based on dataset
                    legend_title = filename
                    if filename == "DNI.tif":
                        legend_title = "Solar Irradiance (kWh/m²)"
                    elif filename == "NAM_wind-speed_100m.tif":
                        legend_title = "Wind Speed at 100m (m/s)"
                    elif filename == "NamDEM.tif":
                        legend_title = "Elevation (m)"
                        
                    legend_text = f"{legend_title}\nMin: {vmin:.1f}\nMax: {vmax:.1f}"
                    legend = self.map_widget.set_text(
                        (top + bottom) / 2, left - 0.1,  # Position legend to the left of the raster
                        text=legend_text,
                        font=("Arial", 12)
                    )
                    obj_refs.append(legend)
                    
                    self.layer_objects[filename] = obj_refs

            except Exception as e:
                self.status_var.set(f"Error loading {filename}")
                messagebox.showerror("Raster Load Error", f"Failed to load {filename}\n{str(e)}")

if __name__ == "__main__":
    app = MapApp()
    app.mainloop()