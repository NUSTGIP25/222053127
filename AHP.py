import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import rasterio
from rasterio import features, windows
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
import threading
import logging
from functools import partial
from datetime import datetime

os.environ["PROJ_DATA"] = r"C:\Users\USER\Documents\NUST\2025\GIP\Project\rera-python\myenv\Lib\site-packages\pyproj\proj_dir\share\proj"
os.environ["GDAL_DATA"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "share", "gdal") 
os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "share", "proj")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = "data"
OUTPUT_TIF = "suitability.tif"
AHP_SCALE_MAX = 9  # Standard AHP scale is 1-9
CHUNK_SIZE = 1024  # For processing large rasters in chunks

# Predefined importance presets for different project types
AHP_PRESETS = {
    "Solar Farm": {
        "DNI.tif": {
            "Land Tenure.geojson": 5,
            "Major Rivers.geojson": 5,
            "NamDEM.tif": 4,
            "NamRoads.geojson": 6,
            "NAM_wind-speed_100m.tif": 7,
            "Perennial Catchment Areas.geojson": 3,
            "Powerlines.geojson": 7,
            "R.E. Sources.geojson": 4,
            "Substations.geojson": 7,
        },
        "Land Tenure.geojson": {
            "Major Rivers.geojson": 2,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 3,
            "NAM_wind-speed_100m.tif": 5,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 4,
            "R.E. Sources.geojson": 3,
            "Substations.geojson": 4,
        },
        "Major Rivers.geojson": {
            "NamDEM.tif": 1,
            "NamRoads.geojson": 2,
            "NAM_wind-speed_100m.tif": 3,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamDEM.tif": {
            "NamRoads.geojson": 2,
            "NAM_wind-speed_100m.tif": 2,
            "Perennial Catchment Areas.geojson": 1,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamRoads.geojson": {
            "NAM_wind-speed_100m.tif": 2,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 4,
            "R.E. Sources.geojson": 3,
            "Substations.geojson": 4,
        },
        "NAM_wind-speed_100m.tif": {
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "Perennial Catchment Areas.geojson": {
            "Powerlines.geojson": 2,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 2,
        },
        "Powerlines.geojson": {
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "R.E. Sources.geojson": {
            "Substations.geojson": 2,
        }
    },

    "Wind Farm": {
        "NAM_wind-speed_100m.tif": {
            "DNI.tif": 6,
            "Land Tenure.geojson": 5,
            "Major Rivers.geojson": 3,
            "NamDEM.tif": 4,
            "NamRoads.geojson": 4,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 6,
            "R.E. Sources.geojson": 3,
            "Substations.geojson": 6,
        },
        "DNI.tif": {
            "Land Tenure.geojson": 2,
            "Major Rivers.geojson": 2,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 2,
            "Perennial Catchment Areas.geojson": 1,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "Land Tenure.geojson": {
            "Major Rivers.geojson": 2,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 3,
            "Perennial Catchment Areas.geojson": 1,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "Major Rivers.geojson": {
            "NamDEM.tif": 1,
            "NamRoads.geojson": 2,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamDEM.tif": {
            "NamRoads.geojson": 1,
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamRoads.geojson": {
            "Perennial Catchment Areas.geojson": 2,
            "Powerlines.geojson": 4,
            "R.E. Sources.geojson": 3,
            "Substations.geojson": 4,
        },
        "Perennial Catchment Areas.geojson": {
            "Powerlines.geojson": 2,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 2,
        },
        "Powerlines.geojson": {
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "R.E. Sources.geojson": {
            "Substations.geojson": 2,
        }
    },

    "Green Hydrogen": {
        "DNI.tif": {
            "NAM_wind-speed_100m.tif": 2,
            "Land Tenure.geojson": 3,
            "Major Rivers.geojson": 5,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 4,
            "Perennial Catchment Areas.geojson": 7,
            "Powerlines.geojson": 4,
            "R.E. Sources.geojson": 3,
            "Substations.geojson": 5,
        },
        "NAM_wind-speed_100m.tif": {
            "Land Tenure.geojson": 2,
            "Major Rivers.geojson": 3,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 3,
            "Perennial Catchment Areas.geojson": 5,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 4,
        },
        "Land Tenure.geojson": {
            "Major Rivers.geojson": 4,
            "NamDEM.tif": 2,
            "NamRoads.geojson": 3,
            "Perennial Catchment Areas.geojson": 6,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "Major Rivers.geojson": {
            "NamDEM.tif": 2,
            "NamRoads.geojson": 2,
            "Perennial Catchment Areas.geojson": 5,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamDEM.tif": {
            "NamRoads.geojson": 2,
            "Perennial Catchment Areas.geojson": 4,
            "Powerlines.geojson": 2,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "NamRoads.geojson": {
            "Perennial Catchment Areas.geojson": 3,
            "Powerlines.geojson": 3,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "Perennial Catchment Areas.geojson": {
            "Powerlines.geojson": 2,
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 2,
        },
        "Powerlines.geojson": {
            "R.E. Sources.geojson": 2,
            "Substations.geojson": 3,
        },
        "R.E. Sources.geojson": {
            "Substations.geojson": 2,
        }
    }
}

class AHPWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RERA Version 2.0 (AHP Suitability Analysis)")
        self.geometry("1200x700")
        self.resizable(True, True)
        
        # Initialize data directory
        self.data_dir = DEFAULT_DATA_DIR
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Menu Bar
        self._create_menu()
        
        # Create tabs for different sections
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Datasets tab
        self.datasets_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.datasets_frame, text="Datasets")
        self._setup_datasets_tab()
        
        # AHP Matrix tab
        self.matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_frame, text="AHP Matrix")
        
        # Output options tab
        self.output_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_frame, text="Output Settings")
        self._setup_output_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize variables
        self.datasets = []
        self.entries = {}
        self.dataset_settings = {}  # For storing normalization settings
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(self, variable=self.progress_var)
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # Load datasets from default directory
        self._load_datasets()

    def _create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Data Directory", command=self._select_data_dir)
        file_menu.add_command(label="Load Datasets", command=self._load_datasets)
        file_menu.add_separator()
        file_menu.add_command(label="Export Matrix to CSV", command=self._export_matrix_csv)
        file_menu.add_command(label="Import Matrix from CSV", command=self._import_matrix_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About AHP", command=self._show_about)

    def _setup_datasets_tab(self):
        # Create frames
        control_frame = ttk.Frame(self.datasets_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Dataset list with settings
        self.dataset_list_frame = ttk.LabelFrame(self.datasets_frame, text="Available Datasets")
        self.dataset_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollable frame for datasets
        canvas = tk.Canvas(self.dataset_list_frame)
        scrollbar = ttk.Scrollbar(self.dataset_list_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _setup_output_tab(self):
        # Output settings
        settings_frame = ttk.LabelFrame(self.output_frame, text="Output Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10, anchor=tk.N)
        
        # Output filename
        ttk.Label(settings_frame, text="Output Filename:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_var = tk.StringVar(value=OUTPUT_TIF)
        ttk.Entry(settings_frame, textvariable=self.output_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        # Export/Import buttons
        export_frame = ttk.LabelFrame(self.output_frame, text="Matrix Export/Import")
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(export_frame, text="Export Matrix to CSV", command=self._export_matrix_csv).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Import Matrix from CSV", command=self._import_matrix_csv).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Run button
        run_frame = ttk.Frame(self.output_frame)
        run_frame.pack(fill=tk.X, padx=10, pady=20)
        ttk.Button(run_frame, text="Run Weighted Analysis", command=self._run_analysis).pack(pady=10)

    def _export_matrix_csv(self):
        """Export the current matrix and datasets to a CSV file"""
        try:
            if not self.datasets:
                messagebox.showwarning("No Datasets", "No datasets loaded to export")
                return
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Matrix as CSV"
            )
            
            if not filename:
                return
            
            # Prepare data for CSV export
            export_data = []
            
            # Add metadata
            export_data.append(["# AHP Matrix Export"])
            export_data.append(["# Generated on:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            export_data.append(["# Number of datasets:", len(self.datasets)])
            export_data.append([])  # Empty row
            
            # Add dataset information
            export_data.append(["# Dataset Settings"])
            export_data.append(["Dataset", "Type", "Settings"])
            
            for dataset in self.datasets:
                dataset_type = "Raster" if dataset.endswith('.tif') else "Vector"
                settings = ""
                
                if dataset in self.dataset_settings:
                    if dataset.endswith('.tif'):
                        invert = self.dataset_settings[dataset]['invert'].get()
                        nodata = self.dataset_settings[dataset]['nodata'].get()
                        settings = f"Invert: {invert}, NoData: {nodata}"
                    elif dataset.endswith('.geojson'):
                        decay = self.dataset_settings[dataset]['decay'].get()
                        settings = f"Decay: {decay}"
                
                export_data.append([dataset, dataset_type, settings])
            
            export_data.append([])  # Empty row
            
            # Add pairwise comparison matrix
            export_data.append(["# Pairwise Comparison Matrix"])
            
            # Matrix header
            header = ["Criteria"] + self.datasets
            export_data.append(header)
            
            # Matrix values
            for i, dataset1 in enumerate(self.datasets):
                row = [dataset1]
                for j, dataset2 in enumerate(self.datasets):
                    if i == j:
                        row.append(1)  # Diagonal
                    elif i < j:
                        # Upper triangle - get user input
                        try:
                            value = self.entries[(i, j)][1].get()
                            row.append(float(value) if value else "")
                        except (KeyError, ValueError, TypeError):
                            row.append("")
                    else:
                        # Lower triangle - calculate reciprocal
                        try:
                            upper_value = self.entries[(j, i)][1].get()
                            if upper_value:
                                reciprocal = round(1.0 / float(upper_value), 4)
                                row.append(reciprocal)
                            else:
                                row.append("")
                        except (KeyError, ValueError, TypeError):
                            row.append("")
                
                export_data.append(row)
            
            # Add weights if available
            try:
                matrix = self._get_comparison_matrix()
                if matrix is not None:
                    # Calculate weights
                    evals, evecs = np.linalg.eig(matrix)
                    max_index = np.argmax(evals.real)
                    weights = evecs[:, max_index].real
                    weights = weights / np.sum(weights)  # Normalize
                    
                    export_data.append([])  # Empty row
                    export_data.append(["# Calculated Weights"])
                    export_data.append(["Dataset", "Weight"])
                    
                    for i, dataset in enumerate(self.datasets):
                        export_data.append([dataset, round(weights[i], 6)])
            except Exception as e:
                logger.warning(f"Could not calculate weights for export: {e}")
            
            # Write to CSV
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False, header=False, encoding='utf-8')
            
            messagebox.showinfo("Export Successful", f"Matrix exported to:\n{filename}")
            self.status_var.set(f"Matrix exported to {os.path.basename(filename)}")
            
        except Exception as e:
            logger.error(f"Error exporting matrix: {e}")
            messagebox.showerror("Export Error", f"Failed to export matrix:\n{e}")

    def _import_matrix_csv(self):
        """Import matrix values from a CSV file"""
        try:
            # Ask user for file location
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Import Matrix from CSV"
            )
            
            if not filename:
                return
            
            # Read CSV file
            df = pd.read_csv(filename, header=None, encoding='utf-8')
            
            # Find the pairwise comparison matrix section
            matrix_start_row = None
            for i, row in df.iterrows():
                if len(row) > 0 and str(row[0]).startswith("# Pairwise Comparison Matrix"):
                    matrix_start_row = i + 2  # Skip header comment and column headers
                    break
            
            if matrix_start_row is None:
                messagebox.showerror("Import Error", "Could not find pairwise comparison matrix in CSV file")
                return
            
            # Extract matrix data
            imported_datasets = []
            imported_matrix = {}
            
            for i in range(matrix_start_row, len(df)):
                row_data = df.iloc[i]
                if pd.isna(row_data[0]) or row_data[0] == "":
                    break  # End of matrix
                
                dataset_name = str(row_data[0])
                imported_datasets.append(dataset_name)
                
                # Extract comparison values for this row
                for j in range(1, len(row_data)):
                    if j <= len(imported_datasets):
                        value = row_data[j]
                        if not pd.isna(value) and value != "":
                            try:
                                imported_matrix[(len(imported_datasets)-1, j-1)] = float(value)
                            except (ValueError, TypeError):
                                pass
            
            # Check if imported datasets match current datasets
            if set(imported_datasets) != set(self.datasets):
                messagebox.showwarning("Dataset Mismatch", 
                    "The datasets in the CSV file don't match the currently loaded datasets.\n" +
                    "Please ensure the same datasets are loaded before importing.")
                return
            
            # Create mapping between dataset orders
            dataset_mapping = {}
            for i, imported_dataset in enumerate(imported_datasets):
                if imported_dataset in self.datasets:
                    dataset_mapping[i] = self.datasets.index(imported_dataset)
            
            # Apply imported values to the matrix
            imported_count = 0
            for (i, j), value in imported_matrix.items():
                if i in dataset_mapping and j in dataset_mapping:
                    mapped_i = dataset_mapping[i]
                    mapped_j = dataset_mapping[j]
                    
                    # Only update upper triangle entries
                    if mapped_i < mapped_j:
                        if (mapped_i, mapped_j) in self.entries:
                            self.entries[(mapped_i, mapped_j)][1].set(str(value))
                            self._update_reciprocal(mapped_i, mapped_j)
                            imported_count += 1
                    elif mapped_i > mapped_j:
                        if (mapped_j, mapped_i) in self.entries:
                            # This is a reciprocal value, calculate the original
                            original_value = 1.0 / value if value != 0 else ""
                            self.entries[(mapped_j, mapped_i)][1].set(str(original_value))
                            self._update_reciprocal(mapped_j, mapped_i)
                            imported_count += 1
            
            messagebox.showinfo("Import Successful", 
                f"Successfully imported {imported_count} matrix values from:\n{os.path.basename(filename)}")
            self.status_var.set(f"Imported matrix from {os.path.basename(filename)}")
            
        except Exception as e:
            logger.error(f"Error importing matrix: {e}")
            messagebox.showerror("Import Error", f"Failed to import matrix:\n{e}")

    def _select_data_dir(self):
        new_dir = filedialog.askdirectory(title="Select Data Directory")
        if new_dir:
            self.data_dir = new_dir
            self._load_datasets()
            self.status_var.set(f"Data directory set to: {self.data_dir}")

    def _load_datasets(self):
        try:
            # Clear existing datasets
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.datasets = []
            self.dataset_settings = {}
            
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                self.status_var.set(f"Created data directory: {self.data_dir}")
                return
                
            # Find all .tif and .geojson files
            files = [f for f in os.listdir(self.data_dir) if f.endswith((".tif", ".geojson"))]
            if not files:
                ttk.Label(self.scrollable_frame, text="No compatible files found. Add .tif or .geojson files.").pack(pady=10)
                return
                
            # Add datasets to list with settings
            for i, fname in enumerate(files):
                self.datasets.append(fname)
                frame = ttk.Frame(self.scrollable_frame)
                frame.pack(fill=tk.X, pady=2)
                
                # Dataset name
                ttk.Label(frame, text=fname, width=25, anchor=tk.W).grid(row=0, column=0, padx=5)
                
                # Settings depend on file type
                if fname.endswith('.tif'):
                    # For rasters - normalization direction
                    invert_var = tk.BooleanVar(value=False)
                    ttk.Checkbutton(frame, text="Invert Values", variable=invert_var).grid(row=0, column=1)
                    
                    # Try to get nodata value
                    nodata_var = tk.StringVar(value="")
                    try:
                        with rasterio.open(os.path.join(self.data_dir, fname)) as src:
                            if src.nodata is not None:
                                nodata_var.set(str(src.nodata))
                    except Exception as e:
                        logger.warning(f"Could not read nodata value for {fname}: {e}")
                    
                    ttk.Label(frame, text="NoData Value:").grid(row=0, column=2, padx=(10,0))
                    ttk.Entry(frame, textvariable=nodata_var, width=10).grid(row=0, column=3)
                    
                    self.dataset_settings[fname] = {
                        'invert': invert_var,
                        'nodata': nodata_var
                    }
                elif fname.endswith('.geojson'):
                    # For vectors - distance decay function
                    decay_var = tk.StringVar(value="Linear")
                    ttk.Combobox(frame, textvariable=decay_var, 
                                values=["Linear", "Squared", "Exponential"], 
                                state="readonly", width=15).grid(row=0, column=1)
                    
                    self.dataset_settings[fname] = {
                        'decay': decay_var
                    }
            
            # Build the comparison matrix when datasets are loaded
            self._build_matrix()
            self.status_var.set(f"Loaded {len(self.datasets)} datasets from {self.data_dir}")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            messagebox.showerror("Error", f"Failed to load datasets: {e}")

    def _build_matrix(self):
        # Clear previous matrix
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
            
        if not self.datasets:
            ttk.Label(self.matrix_frame, text="No datasets loaded. Go to Datasets tab first.").pack(pady=20)
            return
            
        ttk.Label(self.matrix_frame, text="Pairwise Comparison Matrix", font=("Segoe UI", 12, "bold")).pack(pady=5)
        
        # Add preset dropdown and clear button
        preset_frame = ttk.Frame(self.matrix_frame)
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(preset_frame, text="Load Preset Values:").pack(side=tk.LEFT, padx=5)
        self.preset_var = tk.StringVar(value="")
        preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                     values=list(AHP_PRESETS.keys()),
                                     state="readonly", width=20)
        preset_dropdown.pack(side=tk.LEFT, padx=5)
        preset_dropdown.bind("<<ComboboxSelected>>", self._load_preset_values)
        
        ttk.Button(preset_frame, text="Apply Preset", command=self._load_preset_values).pack(side=tk.LEFT, padx=5)
        
        # Add Clear Matrix button
        ttk.Button(preset_frame, text="Clear Matrix", command=self._clear_matrix).pack(side=tk.LEFT, padx=5)
        
        # Help text
        help_text = "Enter values from 1-9 where:\n1 = Equal importance\n3 = Moderate importance\n" + \
                    "5 = Strong importance\n7 = Very strong importance\n9 = Extreme importance\n" + \
                    "2,4,6,8 = Intermediate values"
        ttk.Label(self.matrix_frame, text=help_text, justify=tk.LEFT).pack(pady=5)
        
        # Create scrollable frame for matrix
        canvas = tk.Canvas(self.matrix_frame)
        scrollbar_y = ttk.Scrollbar(self.matrix_frame, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(self.matrix_frame, orient="horizontal", command=canvas.xview)
        matrix_frame = ttk.Frame(canvas)
        
        matrix_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=matrix_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Header row and column
        ttk.Label(matrix_frame, text="", width=20).grid(row=0, column=0)
        for i, name in enumerate(self.datasets):
            ttk.Label(matrix_frame, text=name, width=20, anchor="center").grid(row=0, column=i+1)
            ttk.Label(matrix_frame, text=name, width=20, anchor="center").grid(row=i+1, column=0)

        # Matrix cells
        self.entries = {}
        for i in range(len(self.datasets)):
            for j in range(len(self.datasets)):
                if i == j:
                    # Diagonal - always 1
                    lbl = ttk.Label(matrix_frame, text="1", width=10, background="#e0e0e0", anchor="center")
                    lbl.grid(row=i+1, column=j+1, padx=1, pady=1)
                elif j > i:
                    # Upper triangle - user inputs
                    var = tk.StringVar(value="")
                    ent = ttk.Entry(matrix_frame, width=10, justify="center", textvariable=var)
                    ent.grid(row=i+1, column=j+1, padx=1, pady=1)
                    
                    # Validation
                    var.trace_add("write", lambda n, i, m, r=i, c=j, v=var: self._validate_entry(r, c, v))
                    ent.bind("<FocusOut>", lambda e, r=i, c=j: self._update_reciprocal(r, c))
                    
                    self.entries[(i, j)] = (ent, var)
                else:
                    # Lower triangle - reciprocals
                    lbl = ttk.Label(matrix_frame, text="", width=10, background="#f0f0f0", anchor="center")
                    lbl.grid(row=i+1, column=j+1, padx=1, pady=1)
                    self.entries[(i, j)] = lbl
        
        # Consistency check button
        ttk.Button(self.matrix_frame, text="Check Consistency Ratio", 
                  command=self._check_consistency).pack(pady=10)

    def _clear_matrix(self):
        """Clear all values in the pairwise comparison matrix"""
        try:
            if not self.datasets:
                messagebox.showwarning("No Matrix", "No matrix to clear")
                return
            
            # Ask for confirmation
            result = messagebox.askyesno("Clear Matrix", 
                                       "Are you sure you want to clear all matrix values?\n" +
                                       "This action cannot be undone.")
            
            if not result:
                return
            
            # Clear all upper triangle entries (user inputs)
            cleared_count = 0
            for i in range(len(self.datasets)):
                for j in range(i+1, len(self.datasets)):
                    if (i, j) in self.entries:
                        # Clear the entry field
                        self.entries[(i, j)][1].set("")
                        # Clear the corresponding reciprocal label
                        if (j, i) in self.entries:
                            self.entries[(j, i)].config(text="")
                        cleared_count += 1
            
            self.status_var.set(f"Cleared {cleared_count} matrix values")
            messagebox.showinfo("Matrix Cleared", f"Successfully cleared all {cleared_count} matrix values")
            
        except Exception as e:
            logger.error(f"Error clearing matrix: {e}")
            messagebox.showerror("Clear Error", f"Failed to clear matrix: {e}")

    def _load_preset_values(self, event=None):
        """Load preset values into the matrix"""
        preset_name = self.preset_var.get()
        if not preset_name or preset_name not in AHP_PRESETS:
            return
            
        preset = AHP_PRESETS[preset_name]
        
        # Check if all datasets from the preset are in the loaded datasets
        missing_datasets = []
        for dataset in preset:
            if dataset not in self.datasets:
                missing_datasets.append(dataset)
                
        for parent_dataset, comparisons in preset.items():
            for compared_dataset in comparisons:
                if compared_dataset not in self.datasets:
                    missing_datasets.append(compared_dataset)
                    
        missing_datasets = list(set(missing_datasets))  # Remove duplicates
        
        if missing_datasets:
            messagebox.showwarning("Missing Datasets", 
                                  f"Some datasets required for this preset are missing:\n" + 
                                  "\n".join(missing_datasets))
            return
            
        # Apply preset values to the matrix
        for i, dataset1 in enumerate(self.datasets):
            for j, dataset2 in enumerate(self.datasets):
                if i < j:  # Upper triangle
                    try:
                        if dataset1 in preset and dataset2 in preset[dataset1]:
                            value = preset[dataset1][dataset2]
                            self.entries[(i, j)][1].set(str(value))
                            self._update_reciprocal(i, j)
                    except Exception as e:
                        logger.error(f"Error applying preset value for {dataset1} vs {dataset2}: {e}")
                    
        self.status_var.set(f"Applied preset values for {preset_name}")
        messagebox.showinfo("Preset Applied", f"Applied importance values for {preset_name}")

    def _validate_entry(self, row, col, var):
        """Validate input is a number between 1/9 and 9"""
        value = var.get()
        if not value:  # Allow empty string
            return
            
        try:
            num = float(value)
            # Check if in valid AHP range (1/9 to 9)
            if not (1/AHP_SCALE_MAX <= num <= AHP_SCALE_MAX):
                var.set("")
                messagebox.showwarning("Invalid Value", 
                    f"Value must be between {1/AHP_SCALE_MAX:.2f} and {AHP_SCALE_MAX}")
        except ValueError:
            var.set("")
            messagebox.showwarning("Invalid Value", "Please enter a valid number")

    def _update_reciprocal(self, row, col):
        """Update reciprocal value when user enters a value"""
        try:
            value = self.entries[(row, col)][1].get()
            if not value:
                self.entries[(col, row)].config(text="")
                return
                
            val = float(value)
            if val == 0:
                messagebox.showwarning("Invalid Value", "Value cannot be zero")
                self.entries[(row, col)][1].set("")
                return
                
            reciprocal = round(1.0 / val, 4)
            self.entries[(col, row)].config(text=str(reciprocal))
        except Exception as e:
            logger.error(f"Error updating reciprocal: {e}")
            self.entries[(col, row)].config(text="")

    def _check_consistency(self):
        """Calculate and display consistency ratio"""
        try:
            size = len(self.datasets)
            matrix = self._get_comparison_matrix()
            
            if matrix is None:
                return
                
            # Calculate principal eigenvalue and eigenvector
            evals, evecs = np.linalg.eig(matrix)
            max_index = np.argmax(evals.real)
            max_eval = evals[max_index].real
            
            # Calculate consistency index (CI)
            ci = (max_eval - size) / (size - 1)
            
            # Random index (RI) values for different matrix sizes
            ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 
                        7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
            
            ri = ri_values.get(size, 1.5)  # Default to 1.5 for sizes > 10
            
            # Calculate consistency ratio (CR)
            cr = ci / ri if ri != 0 else 0
            
            message = f"Consistency Ratio (CR): {cr:.4f}\n"
            if cr < 0.1:
                message += "✅ Good! CR < 0.1 indicates consistent judgments."
            else:
                message += "⚠️ Warning! CR > 0.1 suggests inconsistent judgments.\n"
                message += "Consider revising your pairwise comparisons."
                
            messagebox.showinfo("Consistency Check", message)
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            messagebox.showerror("Error", f"Failed to check consistency: {e}")

    def _get_comparison_matrix(self):
        """Get the comparison matrix from UI inputs"""
        size = len(self.datasets)
        
        if size == 0:
            messagebox.showwarning("No Datasets", "No datasets loaded")
            return None
            
        # Check if matrix is complete
        for i in range(size):
            for j in range(i+1, size):  # Check upper triangle
                if not self.entries[(i, j)][1].get():
                    messagebox.showwarning("Incomplete Matrix", 
                                        f"Please complete all comparisons")
                    return None
        
        # Create matrix
        matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if i < j:  # Upper triangle (user inputs)
                    try:
                        matrix[i, j] = float(self.entries[(i, j)][1].get())
                    except (ValueError, TypeError):
                        messagebox.showwarning("Invalid Value", 
                                            f"Invalid value at row {i+1}, column {j+1}")
                        return None
                elif i > j:  # Lower triangle (reciprocals)
                    matrix[i, j] = 1.0 / matrix[j, i]
                    
        return matrix

    def _run_analysis(self):
        """Run the weighted suitability analysis"""
        try:
            # Check for datasets
            if not self.datasets:
                messagebox.showwarning("No Datasets", "No datasets loaded")
                return
                
            # Get matrix and calculate weights
            matrix = self._get_comparison_matrix()
            if matrix is None:
                return
                
            # Calculate weights from comparison matrix
            evals, evecs = np.linalg.eig(matrix)
            max_index = np.argmax(evals.real)
            weights = evecs[:, max_index].real
            weights = weights / np.sum(weights)  # Normalize
            
            # Show weights to user
            weight_message = "Calculated weights:\n\n"
            for i, dataset in enumerate(self.datasets):
                weight_message += f"{dataset}: {weights[i]:.4f}\n"
                
            logger.info(weight_message)
            messagebox.showinfo("Weights Calculated", weight_message)
            
            # Ask confirmation before running the analysis
            if not messagebox.askyesno("Confirm Analysis", 
                                      "Do you want to run the analysis with these weights?"):
                return
                
            # Run analysis in background thread
            self.status_var.set("Starting analysis...")
            self.progress_var.set(0)
            
            # Prepare output path
            output_path = os.path.join(self.data_dir, self.output_var.get())
            
            thread = threading.Thread(target=self._process_analysis, 
                                    args=(self.datasets, weights, self.dataset_settings, output_path))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.status_var.set("Analysis failed")

    def _process_analysis(self, datasets, weights, settings, output_path):
        """Process the weighted overlay analysis in the background"""
        try:
            # Prepare reference raster for alignment
            ref_raster = None
            for dataset in datasets:
                if dataset.endswith('.tif'):
                    ref_raster = os.path.join(self.data_dir, dataset)
                    break
                    
            if not ref_raster:
                raise ValueError("No raster datasets found")
                
            # Open reference raster to get metadata
            with rasterio.open(ref_raster) as ref_src:
                ref_profile = ref_src.profile.copy()
                ref_transform = ref_src.transform
                ref_crs = ref_src.crs
                ref_width = ref_src.width
                ref_height = ref_src.height
                ref_bounds = ref_src.bounds
                
                # Create empty output raster filled with zeros
                result_array = np.zeros((ref_height, ref_width), dtype=np.float32)
                
                # Process each dataset
                for i, dataset in enumerate(datasets):
                    self.status_var.set(f"Processing {dataset}...")
                    self.progress_var.set((i / len(datasets)) * 100)
                    self.update_idletasks()
                    
                    dataset_path = os.path.join(self.data_dir, dataset)
                    
                    if dataset.endswith('.tif'):
                        # Process raster datasets
                        nodata_value = None
                        if dataset in settings and 'nodata' in settings[dataset]:
                            try:
                                nodata_str = settings[dataset]['nodata'].get()
                                if nodata_str:
                                    nodata_value = float(nodata_str)
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid nodata value for {dataset}")
                        
                        # Open raster and align to reference
                        with rasterio.open(dataset_path) as src_ds:
                            # Create destination array with reference dimensions
                            aligned_data = np.zeros((ref_height, ref_width), dtype=np.float32)
                            
                            # Reproject and resample to match reference raster exactly
                            reproject(
                                source=rasterio.band(src_ds, 1),
                                destination=aligned_data,
                                src_transform=src_ds.transform,
                                src_crs=src_ds.crs,
                                dst_transform=ref_transform,
                                dst_crs=ref_crs,
                                resampling=Resampling.bilinear,
                                src_nodata=src_ds.nodata,
                                dst_nodata=np.nan
                            )
                            
                            # Handle custom nodata value if specified
                            if nodata_value is not None:
                                aligned_data = np.where(aligned_data == nodata_value, np.nan, aligned_data)
                            
                            # Normalize data to 0-1 range
                            valid_data = aligned_data[~np.isnan(aligned_data)]
                            if len(valid_data) > 0:
                                min_val = np.min(valid_data)
                                max_val = np.max(valid_data)
                                
                                if min_val != max_val:  # Avoid division by zero
                                    norm_data = (aligned_data - min_val) / (max_val - min_val)
                                    
                                    # Invert if requested
                                    if dataset in settings and settings[dataset]['invert'].get():
                                        norm_data = 1 - norm_data
                                        
                                    # Replace NaN with 0 and add to result with weight
                                    norm_data = np.nan_to_num(norm_data, nan=0)
                                    result_array += norm_data * weights[i]
                                else:
                                    logger.warning(f"Dataset {dataset} has constant values, skipping normalization")
                            else:
                                logger.warning(f"Dataset {dataset} has no valid data")
                    
                    elif dataset.endswith('.geojson'):
                        # Process vector datasets
                        try:
                            # Read vector data
                            gdf = gpd.read_file(dataset_path)
                            
                            # Reproject if necessary
                            if gdf.crs != ref_crs:
                                gdf = gdf.to_crs(ref_crs)
                            
                            # Create raster representation aligned to reference
                            vector_raster = np.zeros((ref_height, ref_width), dtype=np.uint8)
                            
                            # Rasterize vector data using reference transform and dimensions
                            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
                            if shapes:
                                try:
                                    rasterized = features.rasterize(
                                        shapes=shapes,
                                        out_shape=(ref_height, ref_width),
                                        transform=ref_transform,
                                        fill=0,
                                        dtype=np.uint8
                                    )
                                    vector_raster = rasterized
                                except Exception as e:
                                    logger.error(f"Error rasterizing {dataset}: {e}")
                                    continue
                            
                            # Distance transform for proximity analysis
                            if np.any(vector_raster):
                                # Invert to get distance from features
                                binary_mask = vector_raster == 0
                                
                                # Convert pixel size to meters for consistent distance measurement
                                # Assuming the CRS is in a projected coordinate system
                                pixel_width_m = abs(ref_transform[0])
                                
                                # Calculate euclidean distance
                                distances = distance_transform_edt(binary_mask) * pixel_width_m
                                
                                # Get max distance for normalization
                                max_distance = np.max(distances)
                                if max_distance > 0:
                                    # Normalize distances to 0-1
                                    norm_distances = distances / max_distance
                                    
                                    # Apply decay function
                                    decay_type = "Linear"  # Default
                                    if dataset in settings and 'decay' in settings[dataset]:
                                        decay_type = settings[dataset]['decay'].get()
                                    
                                    if decay_type == "Linear":
                                        # Linear decay (1 at feature, 0 at max distance)
                                        norm_distances = 1 - norm_distances
                                    elif decay_type == "Squared":
                                        # Squared decay (faster falloff)
                                        norm_distances = (1 - norm_distances) ** 2
                                    elif decay_type == "Exponential":
                                        # Exponential decay
                                        norm_distances = np.exp(-3 * norm_distances)
                                    
                                    # Add to result with weight
                                    result_array += norm_distances * weights[i]
                                else:
                                    logger.warning(f"No distance variation found for {dataset}")
                            else:
                                logger.warning(f"No features found in {dataset} within the analysis area")
                                
                        except Exception as e:
                            logger.error(f"Error processing vector {dataset}: {e}")
                            continue
                
                # Normalize final result to 0-1 if there's variation
                result_min = np.min(result_array)
                result_max = np.max(result_array)
                
                if result_min != result_max:
                    result_array = (result_array - result_min) / (result_max - result_min)
                else:
                    logger.warning("Result array has no variation - all values are the same")
                
                # Write output raster using reference profile
                output_profile = ref_profile.copy()
                output_profile.update({
                    'dtype': rasterio.float32,
                    'count': 1,
                    'nodata': None
                })
                
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(result_array.astype(rasterio.float32), 1)
                
                self.status_var.set(f"Analysis complete. Output saved to {output_path}")
                self.progress_var.set(100)
                
                # Show success message
                messagebox.showinfo("Analysis Complete", 
                                f"Suitability analysis completed.\nOutput saved to {self.output_var.get()}")
                
        except Exception as e:
            logger.error(f"Error in analysis processing: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.status_var.set("Analysis failed")

    def _show_about(self):
        about_text = """Renewable Energy Resource Assessment (RERA) Version 2.0
        
This tool performs suitability analysis for renewable energy projects using Analytic Hierarchy Process (AHP).

AHP is a structured technique for organizing and analyzing complex decisions. It provides a comprehensive framework for comparing criteria, calculating weights, and performing multi-criteria analysis.

How to use:
1. Load your spatial datasets (.tif rasters and .geojson vectors)
2. Set dataset-specific settings
3. Fill in the pairwise comparison matrix or select a preset
4. Check consistency of your judgments
5. Run the analysis to generate a suitability map

CSV Export/Import:
- Export your matrix to CSV for documentation or sharing
- Import previously saved matrix configurations
- CSV files include dataset information, settings, and calculated weights

© 2025 Jeffrey N.S. Shigwedha (222053127)
"""
        messagebox.showinfo("About AHP Suitability Analysis", about_text)

if __name__ == "__main__":
    app = AHPWindow()
    app.mainloop()