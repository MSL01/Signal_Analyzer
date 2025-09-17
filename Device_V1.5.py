import sys
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QSizePolicy, QLineEdit, QComboBox,
    QCheckBox, QGroupBox, QGridLayout, QScrollArea, QListWidget,
    QListWidgetItem, QAbstractItemView, QColorDialog, QTabWidget
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class SignalAnalyzer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('Signal Analyzer System')
        self.df = None
        self.df_normalized = None
        self.df_filtered = None
        self.annotations = []
        self.markers = []
        self.delta_lines = []
        self.selected_points = []
        self.dragging_annotation = None
        self.drag_offset = (0, 0)
        self.signal_colors = {}
        self.current_ax = None
        self.plotted_signals = {}
        self.plot_map = {}
        self.signal_axes = []
        self.active_plot_channels = set()
        self.plotted_lines = []
        self.is_normalized = False
        self.is_filtered = False
        self.sampling_rate = 100000

        # FFT attributes
        self.fft_figure = None
        self.fft_canvas = None
        self.fft_ax = None
        self.fft_annotations = []
        self.fft_markers = []
        self.fft_selected_points = []
        self.dragging_fft_annotation = None
        self.dragging_fft_index = None
        self.drag_fft_offset = (0, 0)

        # FFT dragging attributes
        self.dragging_fft_annotation = None
        self.dragging_fft_index = None
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_annotation_start_X = None
        self.drag_annotation_start_Y = None

        # CWT attributes
        self.cwt_figure = None
        self.cwt_canvas = None
        self.cwt_ax = None
        self.cwt_annotations = []
        self.cwt_markers = []
        # CWT parameters
        self.wavelet_types = ['mexh', 'morl', 'cgau1',
                              'gaus1', 'shan2.0-2.0',
                              'fbsp1-1.5-1.5',
                              'fbsp2-1.5-1.5', 'fbsp3-1.5-1.5',
                              'cmor1.5-2.0', 'cmor2.0-2.0']
        self.scales_range = [1, 128]

        self.setup()

    def setup(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        """
        --- Top Panel ---
        """
        header_widget = QWidget()
        header_widget.setFixedHeight(35)
        header_widget.setObjectName("top_panel")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 2, 10, 2)
        header_layout.setSpacing(5)

        self.load_btn = QPushButton('📁 Load CSV')
        self.plot_btn = QPushButton('📊 Graph')
        self.clear_btn = QPushButton('🧹 Clear all')
        self.legend_btn = QPushButton('📋 Legend')
        self.normalized_btn = QPushButton('Normalize')
        self.fft_btn = QPushButton('📈 FFT')
        self.cwt_btn = QPushButton('🌊 CWT')

        button_style = """
            QPushButton {
                font-size: 10px; padding: 3px 8px; margin: 0px; 
                border: 1px solid #ccc; background-color: #f8f8f8;
                min-width: 70px; max-height: 25px;
                color: black;
            }
            QPushButton:hover { background-color: #e8e8e8; }
            QPushButton:pressed { background-color: #d8d8d8; }
        """

        for btn in [self.load_btn, self.plot_btn, self.clear_btn, self.normalized_btn, self.fft_btn, self.cwt_btn]:
            btn.setStyleSheet(button_style)

        header_layout.addWidget(self.load_btn)
        header_layout.addWidget(self.plot_btn)
        header_layout.addWidget(self.clear_btn)
        header_layout.addWidget(self.legend_btn)
        header_layout.addWidget(self.normalized_btn)
        header_layout.addWidget(self.fft_btn)
        header_layout.addWidget(self.cwt_btn)
        header_layout.addStretch()
        main_layout.addWidget(header_widget)

        """
        --- Central Panels ---
        """
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)

        """
        --- Left Panel ---
        """
        left_panel = QWidget()
        left_panel.setMinimumWidth(450)
        left_panel.setObjectName("left_panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        """
        Data Table
        """
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.data_table.setMaximumHeight(200)
        left_layout.addWidget(self.data_table)

        """
        Axis Selection
        """
        axes_group = QGroupBox("Axis Selection")
        axes_layout = QGridLayout(axes_group)

        axes_layout.addWidget(QLabel("Axis X:"), 0, 0)
        self.x_axis_combo = QComboBox()
        axes_layout.addWidget(self.x_axis_combo, 0, 1)

        axes_layout.addWidget(QLabel("Range X:"), 1, 0)
        self.x_range_layout = QHBoxLayout()
        self.x_min_input = QLineEdit("0")
        self.x_min_input.setPlaceholderText("Min")
        self.x_max_input = QLineEdit("")
        self.x_max_input.setPlaceholderText("Max (Empty = all data)")
        self.x_range_layout.addWidget(self.x_min_input)
        self.x_range_layout.addWidget(QLabel("a"))
        self.x_range_layout.addWidget(self.x_max_input)
        axes_layout.addLayout(self.x_range_layout, 1, 1)

        left_layout.addWidget(axes_group)

        """
        Y-axis Column Selector
        """
        y_columns_group = QGroupBox("Signals to Graph (Axis Y)")
        y_columns_layout = QVBoxLayout(y_columns_group)
        self.y_columns_list = QListWidget()
        self.y_columns_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.y_columns_list.itemDoubleClicked.connect(self.change_signal_color)
        y_columns_layout.addWidget(self.y_columns_list)
        left_layout.addWidget(y_columns_group)

        """
        Controls Tabs
        """
        controls_tabs = QTabWidget()

        """
        Tab 1: Graph Controls
        """
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)

        plot_controls_group = QGroupBox("Graph Parameters")
        plot_controls_layout = QGridLayout(plot_controls_group)

        plot_controls_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_input = QLineEdit("Signal plotter")
        plot_controls_layout.addWidget(self.title_input, 0, 1)

        plot_controls_layout.addWidget(QLabel("Axis X:"), 1, 0)
        self.x_label_input = QLineEdit("Time (s)")
        plot_controls_layout.addWidget(self.x_label_input, 1, 1)

        plot_controls_layout.addWidget(QLabel("Axis Y:"), 2, 0)
        self.y_label_input = QLineEdit("Voltage (V)")
        plot_controls_layout.addWidget(self.y_label_input, 2, 1)

        plot_controls_layout.addWidget(QLabel("Set Distance:"), 3, 0)
        self.distance_input = QLineEdit('1')
        plot_controls_layout.addWidget(self.distance_input, 3, 1)

        plot_controls_layout.addWidget(QLabel("Set LineWidth:"), 4, 0)
        self.line_input = QLineEdit('1')
        plot_controls_layout.addWidget(self.line_input, 4, 1)

        graph_layout.addWidget(plot_controls_group)
        graph_layout.addStretch()

        """
        Tab 2: Filters Controls
        """
        filters_tab = QWidget()
        filters_layout = QVBoxLayout(filters_tab)

        filters_scroll = QScrollArea()
        filters_scroll.setWidgetResizable(True)
        filters_scroll.setMaximumHeight(300)

        filters_container = QWidget()
        filters_container_layout = QVBoxLayout(filters_container)

        filters_group = QGroupBox("Filter Settings")
        self.filters_grid = QGridLayout(filters_group)

        self.filters_grid.addWidget(QLabel("Sampling Rate:"), 0, 0)
        self.sampling_rate_input = QLineEdit('100000')
        self.filters_grid.addWidget(self.sampling_rate_input, 0, 1)
        self.filters_grid.addWidget(QLabel("Hz"), 0, 2)

        self.filters_grid.addWidget(QLabel("Filter Type:"), 1, 0)
        self.filter_type = QComboBox()
        self.filter_type.addItems(["None", "Low-pass", "High-pass", "Band-pass", "Notch", "Band-stop"])
        self.filter_type.currentTextChanged.connect(self.update_filter_parameters)
        self.filters_grid.addWidget(self.filter_type, 1, 1, 1, 2)

        self.filters_grid.addWidget(QLabel("Filter Order:"), 2, 0)
        self.filter_order = QLineEdit('4')
        self.filters_grid.addWidget(self.filter_order, 2, 1, 1, 2)

        self.filter_params_container = QWidget()
        self.filter_params_layout = QGridLayout(self.filter_params_container)
        self.filters_grid.addWidget(self.filter_params_container, 3, 0, 1, 3)

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.filters_grid.addWidget(self.apply_filter_btn, 4, 0, 1, 3)

        self.remove_filter_btn = QPushButton("Remove Filter")
        self.filters_grid.addWidget(self.remove_filter_btn, 5, 0, 1, 3)

        filters_container_layout.addWidget(filters_group)
        filters_container_layout.addStretch()

        filters_scroll.setWidget(filters_container)
        filters_layout.addWidget(filters_scroll)

        """
        Tab 3: FFT Controls
        """
        fft_tab = QWidget()
        fft_layout = QVBoxLayout(fft_tab)

        fft_scroll = QScrollArea()
        fft_scroll.setWidgetResizable(True)
        fft_scroll.setMaximumHeight(300)

        fft_container = QWidget()
        fft_container_layout = QVBoxLayout(fft_container)

        fft_group = QGroupBox("FFT Settings")
        fft_grid = QGridLayout(fft_group)

        # Sampling rate for FFT
        fft_grid.addWidget(QLabel("Sampling Rate:"), 0, 0)
        self.fft_sampling_rate = QLineEdit('100000')
        fft_grid.addWidget(self.fft_sampling_rate, 0, 1)
        fft_grid.addWidget(QLabel("Hz"), 0, 2)

        # Window type selection
        fft_grid.addWidget(QLabel("Window Type:"), 1, 0)
        self.window_type = QComboBox()
        self.window_type.addItems(["Rectangular", "Hanning", "Hamming", "Blackman", "Bartlett", "Kaiser"])
        fft_grid.addWidget(self.window_type, 1, 1, 1, 2)

        # Kaiser beta parameter (only visible for Kaiser window)
        fft_grid.addWidget(QLabel("Kaiser Beta:"), 2, 0)
        self.kaiser_beta = QLineEdit('14')
        fft_grid.addWidget(self.kaiser_beta, 2, 1, 1, 2)
        self.kaiser_beta_label = QLabel("Beta:")
        self.kaiser_beta_input = QLineEdit('14')
        fft_grid.addWidget(self.kaiser_beta_label, 2, 0)
        fft_grid.addWidget(self.kaiser_beta_input, 2, 1)
        fft_grid.addWidget(QLabel(""), 2, 2)
        self.kaiser_beta_label.setVisible(False)
        self.kaiser_beta_input.setVisible(False)

        # Zero padding
        fft_grid.addWidget(QLabel("Zero Padding:"), 3, 0)
        self.zero_padding = QComboBox()
        self.zero_padding.addItems(["None", "2x", "4x", "8x", "Custom"])
        fft_grid.addWidget(self.zero_padding, 3, 1, 1, 2)

        # Custom zero padding factor
        fft_grid.addWidget(QLabel("Custom Padding:"), 4, 0)
        self.custom_padding = QLineEdit('1024')
        fft_grid.addWidget(self.custom_padding, 4, 1, 1, 2)
        self.custom_padding_label = QLabel("Custom Padding:")
        self.custom_padding_input = QLineEdit('1024')
        fft_grid.addWidget(self.custom_padding_label, 4, 0)
        fft_grid.addWidget(self.custom_padding_input, 4, 1)
        fft_grid.addWidget(QLabel(""), 4, 2)
        self.custom_padding_label.setVisible(False)
        self.custom_padding_input.setVisible(False)

        # Smoothing options
        fft_grid.addWidget(QLabel("Smoothing:"), 5, 0)
        self.smoothing_type = QComboBox()
        self.smoothing_type.addItems(["None", "Moving Average", "Savitzky-Golay"])
        fft_grid.addWidget(self.smoothing_type, 5, 1, 1, 2)

        # Smoothing window size
        fft_grid.addWidget(QLabel("Smoothing Window:"), 6, 0)
        self.smoothing_window = QLineEdit('11')
        fft_grid.addWidget(self.smoothing_window, 6, 1, 1, 2)
        self.smoothing_window_label = QLabel("Window Size:")
        self.smoothing_window_input = QLineEdit('11')
        fft_grid.addWidget(self.smoothing_window_label, 6, 0)
        fft_grid.addWidget(self.smoothing_window_input, 6, 1)
        fft_grid.addWidget(QLabel(""), 6, 2)
        self.smoothing_window_label.setVisible(False)
        self.smoothing_window_input.setVisible(False)

        # Frequency range
        fft_grid.addWidget(QLabel("Freq Range Min:"), 7, 0)
        self.freq_min = QLineEdit('0')
        fft_grid.addWidget(self.freq_min, 7, 1)
        fft_grid.addWidget(QLabel("Hz"), 7, 2)

        fft_grid.addWidget(QLabel("Freq Range Max:"), 8, 0)
        self.freq_max = QLineEdit('25')
        fft_grid.addWidget(self.freq_max, 8, 1)
        fft_grid.addWidget(QLabel("Hz"), 8, 2)

        # Y-axis scale
        fft_grid.addWidget(QLabel("Y Scale:"), 9, 0)
        self.y_scale = QComboBox()
        self.y_scale.addItems(["Linear", "Logarithmic"])
        fft_grid.addWidget(self.y_scale, 9, 1, 1, 2)

        # Normalization
        fft_grid.addWidget(QLabel("Normalization:"), 10, 0)
        self.fft_normalization = QComboBox()
        self.fft_normalization.addItems(["None", "Amplitude", "Power", "PSD"])
        fft_grid.addWidget(self.fft_normalization, 10, 1, 1, 2)

        # Apply FFT button
        self.apply_fft_btn = QPushButton("Apply FFT")
        fft_grid.addWidget(self.apply_fft_btn, 11, 0, 1, 3)

        # Connect signals for dynamic UI updates
        self.window_type.currentTextChanged.connect(self.update_fft_ui)
        self.zero_padding.currentTextChanged.connect(self.update_fft_ui)
        self.smoothing_type.currentTextChanged.connect(self.update_fft_ui)

        fft_container_layout.addWidget(fft_group)
        fft_container_layout.addStretch()

        fft_scroll.setWidget(fft_container)
        fft_layout.addWidget(fft_scroll)

        """
        Tab 4: CWT Controls
        """
        cwt_tab = QWidget()
        cwt_layout = QVBoxLayout(cwt_tab)

        cwt_scroll = QScrollArea()
        cwt_scroll.setWidgetResizable(True)
        cwt_scroll.setMaximumHeight(300)

        cwt_container = QWidget()
        cwt_container_layout = QVBoxLayout(cwt_container)

        cwt_group = QGroupBox("CWT Settings")
        cwt_grid = QGridLayout(cwt_group)

        # Wavelet selection
        cwt_grid.addWidget(QLabel("Wavelet Type:"), 0, 0)
        self.wavelet_type = QComboBox()
        self.wavelet_type.addItems(self.wavelet_types)
        cwt_grid.addWidget(self.wavelet_type, 0, 1, 1, 2)

        # Scales range
        cwt_grid.addWidget(QLabel("Scales Range:"), 1, 0)
        self.scales_min = QLineEdit('1')
        self.scales_max = QLineEdit('128')
        scales_range_layout = QHBoxLayout()
        scales_range_layout.addWidget(self.scales_min)
        scales_range_layout.addWidget(QLabel("to"))
        scales_range_layout.addWidget(self.scales_max)
        cwt_grid.addLayout(scales_range_layout, 1, 1, 1, 2)

        # Number of scales
        cwt_grid.addWidget(QLabel("Number of Scales:"), 2, 0)
        self.num_scales = QLineEdit('64')
        cwt_grid.addWidget(self.num_scales, 2, 1, 1, 2)

        # Color limits
        cwt_grid.addWidget(QLabel("Color Min:"), 3, 0)
        self.cwt_min = QLineEdit('0')
        cwt_grid.addWidget(self.cwt_min, 3, 1)

        cwt_grid.addWidget(QLabel("Color Max:"), 4, 0)
        self.cwt_max = QLineEdit('100')
        cwt_grid.addWidget(self.cwt_max, 4, 1)

        # Sampling rate for CWT
        cwt_grid.addWidget(QLabel("Sampling Rate:"), 5, 0)
        self.cwt_sampling_rate = QLineEdit('100000')
        cwt_grid.addWidget(self.cwt_sampling_rate, 5, 1)
        cwt_grid.addWidget(QLabel("Hz"), 5, 2)

        # Colormap selection
        cwt_grid.addWidget(QLabel("Colormap:"), 6, 0)
        self.colormap = QComboBox()
        self.colormap.addItems(
            ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'])
        cwt_grid.addWidget(self.colormap, 6, 1, 1, 2)

        # View mode selection
        cwt_grid.addWidget(QLabel("View Mode:"), 7, 0)
        self.cwt_view_mode = QComboBox()
        self.cwt_view_mode.addItems(["Separate Subplots", "Overlay", "Single Plot"])
        cwt_grid.addWidget(self.cwt_view_mode, 7, 1, 1, 2)

        # Apply CWT button
        self.apply_cwt_btn = QPushButton("Apply CWT")
        cwt_grid.addWidget(self.apply_cwt_btn, 8, 0, 1, 3)

        cwt_container_layout.addWidget(cwt_group)
        cwt_container_layout.addStretch()

        cwt_scroll.setWidget(cwt_container)
        cwt_layout.addWidget(cwt_scroll)

        # Add all tabs to controls
        controls_tabs.addTab(graph_tab, "Graph Controls")
        controls_tabs.addTab(filters_tab, "Filters Controls")
        controls_tabs.addTab(fft_tab, "FFT Controls")
        controls_tabs.addTab(cwt_tab, "CWT Controls")
        left_layout.addWidget(controls_tabs)

        """
        File Information
        """
        self.file_info_label = QLabel("No CSV Loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
        self.file_info_label.setMaximumHeight(80)
        left_layout.addWidget(self.file_info_label)
        left_layout.addStretch()

        content_splitter.addWidget(left_panel)

        """
        --- Right Panel ---
        """
        right_panel = QWidget()
        right_panel.setObjectName("right_panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        """
        Matplotlib Figure with Tabs for Signal, FFT and CWT
        """
        self.plot_tabs = QTabWidget()

        # Signal Tab
        self.signal_tab = QWidget()
        signal_tab_layout = QVBoxLayout(self.signal_tab)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        signal_tab_layout.addWidget(self.toolbar)
        signal_tab_layout.addWidget(self.canvas)

        # FFT Tab
        self.fft_tab = QWidget()
        fft_tab_layout = QVBoxLayout(self.fft_tab)
        self.fft_figure = Figure(figsize=(12, 8))
        self.fft_canvas = FigureCanvas(self.fft_figure)
        self.fft_toolbar = NavigationToolbar(self.fft_canvas, self)
        fft_tab_layout.addWidget(self.fft_toolbar)
        fft_tab_layout.addWidget(self.fft_canvas)

        # CWT Tab
        self.cwt_tab = QWidget()
        cwt_tab_layout = QVBoxLayout(self.cwt_tab)
        self.cwt_figure = Figure(figsize=(12, 8))
        self.cwt_canvas = FigureCanvas(self.cwt_figure)
        self.cwt_toolbar = NavigationToolbar(self.cwt_canvas, self)
        cwt_tab_layout.addWidget(self.cwt_toolbar)
        cwt_tab_layout.addWidget(self.cwt_canvas)

        self.plot_tabs.addTab(self.signal_tab, "Signal")
        self.plot_tabs.addTab(self.fft_tab, "FFT")
        self.plot_tabs.addTab(self.cwt_tab, "CWT")

        right_layout.addWidget(self.plot_tabs)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([450, 1150])

        """
        Signal Connectors
        """
        self.load_btn.clicked.connect(self.load_csv)
        self.plot_btn.clicked.connect(self.plot_signals)
        self.clear_btn.clicked.connect(self.clear_plot)
        self.legend_btn.clicked.connect(self.toggle_legend)
        self.normalized_btn.clicked.connect(self.normalized_signal)
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.remove_filter_btn.clicked.connect(self.remove_filter)
        self.fft_btn.clicked.connect(self.calculate_fft)
        self.cwt_btn.clicked.connect(self.calculate_cwt)
        self.apply_cwt_btn.clicked.connect(self.calculate_cwt)
        self.apply_fft_btn.clicked.connect(self.calculate_fft)
        self.x_axis_combo.currentTextChanged.connect(self.update_y_columns_list)

        """
        Event Connectors
        """
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.fft_canvas.mpl_connect('button_press_event', self.on_fft_click)
        self.fft_canvas.mpl_connect('motion_notify_event', self.on_fft_motion)
        self.fft_canvas.mpl_connect('button_release_event', self.on_fft_release)

        """
        Initialize filter parameters
        """
        self.update_filter_parameters()
        self.update_fft_ui()

    def update_fft_ui(self):
        """Update FFT UI based on selected options"""
        # Show/hide Kaiser beta parameter
        show_kaiser = self.window_type.currentText() == "Kaiser"
        self.kaiser_beta_label.setVisible(show_kaiser)
        self.kaiser_beta_input.setVisible(show_kaiser)

        # Show/hide custom padding parameter
        show_custom_padding = self.zero_padding.currentText() == "Custom"
        self.custom_padding_label.setVisible(show_custom_padding)
        self.custom_padding_input.setVisible(show_custom_padding)

        # Show/hide smoothing window parameter
        show_smoothing = self.smoothing_type.currentText() != "None"
        self.smoothing_window_label.setVisible(show_smoothing)
        self.smoothing_window_input.setVisible(show_smoothing)

    def change_signal_color(self, item):
        column_name = item.text()
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            self.signal_colors[column_name] = hex_color
            item.setBackground(QColor(hex_color))
            item.setForeground(QColor('white') if color.lightness() < 128 else QColor('black'))

    def load_csv(self):
        """
        Seek and open a CSV file
        :return: .csv File
        """
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df_normalized = None  # Resetear datos normalizados
                self.is_normalized = False
                self.show_data(self.df.head(20))
                self.update_column_selectors()
                self.signal_colors = {}
                total_rows = sum(1 for _ in open(file_path)) - 1
                info_text = f"📊 File: {file_path.split('/')[-1]}\n"
                info_text += f"📈 Rows: {total_rows:,}, Columns: {len(self.df.columns)}\n"
                info_text += f"👀 Showing: {min(20, total_rows)} preview"
                self.file_info_label.setText(info_text)
            except Exception as e:
                error_label = QLabel(f"Error: {str(e)}")
                self.bottom_layout.addWidget(error_label)

    def show_data(self, df):
        """
        Show the data set parameters
        :param df: DataFrame Loaded
        """
        rows, cols = df.shape
        self.data_table.setRowCount(rows)
        self.data_table.setColumnCount(cols)
        self.data_table.setHorizontalHeaderLabels(df.columns)
        for i in range(rows):
            for j in range(cols):
                value = str(df.iloc[i, j])
                item = QTableWidgetItem(value)
                if len(value) > 30:
                    item.setText(value[:27] + "...")
                self.data_table.setItem(i, j, item)

        self.data_table.resizeRowsToContents()

    def update_column_selectors(self):
        if self.df is not None:
            self.x_axis_combo.clear()
            self.x_axis_combo.addItems(self.df.columns)
            if len(self.df.columns) > 0:
                self.x_axis_combo.setCurrentIndex(0)
            self.update_y_columns_list()

    def update_y_columns_list(self):
        if self.df is not None:
            self.y_columns_list.clear()
            x_column = self.x_axis_combo.currentText()
            for column in self.df.columns:
                if column != x_column:
                    item = QListWidgetItem(column)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    if column in self.signal_colors:
                        item.setBackground(QColor(self.signal_colors[column]))
                        item.setForeground(QColor('white'))
                    self.y_columns_list.addItem(item)

    def plot_signals(self):
        if self.df is None or self.y_columns_list.count() == 0:
            return
        try:
            self.figure.clear()
            self.clear_plot_elements()
            self.ax = self.figure.add_subplot(111)
            x_column = self.x_axis_combo.currentText()
            selected_y_columns = []
            for i in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_y_columns.append(item.text())
            if not selected_y_columns:
                return

            if self.is_filtered and self.df_filtered is not None:
                plot_data = self.df_filtered
                y_label = "Voltage (V)"
            elif self.is_normalized and self.df_normalized is not None:
                plot_data = self.df_normalized
                y_label = "Voltage (V)"
            else:
                plot_data = self.df
                y_label = self.y_label_input.text()

            x_min = self.x_min_input.text().strip()
            x_max = self.x_max_input.text().strip()
            plot_data = plot_data.copy()
            if x_min:
                try:
                    x_min_val = float(x_min)
                    plot_data = plot_data[plot_data[x_column] >= x_min_val]
                except ValueError:
                    pass
            if x_max:
                try:
                    x_max_val = float(x_max)
                    plot_data = plot_data[plot_data[x_column] <= x_max_val]
                except ValueError:
                    pass

            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_y_columns)))
            self.plotted_lines = []
            legend_handles = []

            for i, y_column in enumerate(selected_y_columns):
                color = self.signal_colors.get(y_column, colors[i])
                line, = self.ax.plot(plot_data[x_column], plot_data[y_column],
                                     label=y_column, color=color,
                                     linewidth=float(self.line_input.text()),
                                     alpha=0.8)
                legend_handles.append(line)
                self.plotted_lines.append((line, y_column))

            # Set titles and labels
            title = self.title_input.text()
            if self.is_filtered:
                filter_type = self.filter_type.currentText()

            x_label = self.x_label_input.text()

            self.ax.set_xlabel(x_label, fontsize=20)
            self.ax.set_ylabel(y_label, fontsize=20)
            self.ax.set_title(title, fontsize=16, fontweight='bold')
            self.ax.grid(False)
            self.ax.legend(handles=legend_handles, loc='best', framealpha=0.9, fontsize=18)
            self.figure.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error in plotting: {e}")

    def toggle_legend(self):
        if hasattr(self, 'ax') and self.ax.get_legend():
            legend = self.ax.get_legend()
            legend.set_visible(not legend.get_visible())
            self.canvas.draw_idle()

    def clear_plot(self):
        self.figure.clear()
        self.clear_plot_elements()
        self.plotted_lines = []
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.button == 3:
            self.clear_plot_elements()
            self.canvas.draw_idle()
            return

        if event.inaxes != self.ax or event.button != 1:
            return

        if self.toolbar.mode != '':
            return

        for annotation in self.annotations:
            if annotation.contains(event)[0]:
                self.dragging_annotation = annotation
                x_ann, y_ann = annotation.get_position()
                self.drag_offset = (event.xdata - x_ann, event.ydata - y_ann)
                return

        x_click = event.xdata
        y_click = event.ydata
        if x_click is None or y_click is None:
            return

        min_dist = float('inf')
        closest_point = None
        closest_line = None

        try:
            distance_value = float(self.distance_input.text())
        except ValueError:
            distance_value = 1.0

        for line, column_name in self.plotted_lines:
            x_values = line.get_xdata()
            y_values = line.get_ydata()

            idx = np.abs(x_values - x_click).argmin()
            dist = np.hypot(x_click - x_values[idx], y_click - y_values[idx])

            if dist < min_dist:
                min_dist = dist
                closest_point = (x_values[idx], y_values[idx])
                closest_line = line

        if closest_point:
            x_real, y_real = closest_point

            marker_color = closest_line.get_color()

            marker, = self.ax.plot(x_real, y_real, marker='o', color=marker_color,
                                   markersize=10, zorder=5, alpha=0.8)
            self.markers.append(marker)
            self.selected_points.append((x_real, y_real))

            if len(self.selected_points) == 2:
                (x1, y1), (x2, y2) = self.selected_points
                linea, = self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=4, alpha=0.7)
                self.delta_lines.append(linea)

                dx = x2 - x1
                dy = y2 - y1

                texto = (f"P1: (t={x1:.6f} s)\n"
                         f"P2: (t={x2:.6f} s)\n"
                         f"Δt = {dx:.6f} s\n"
                         f"v = {np.abs(distance_value / dx):.6f} m/s")

                y_pos = 0.95 - len(self.annotations) * 0.12

                anotacion = self.ax.annotate(
                    texto,
                    xy=((x1 + x2) / 2, (y1 + y2) / 2),
                    xycoords='data',
                    xytext=(0.02, y_pos),
                    textcoords='axes fraction',
                    fontsize=12,
                    va='top',
                    ha='left',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black')
                )
                self.annotations.append(anotacion)
                self.selected_points = []

            self.canvas.draw_idle()

    def on_motion(self, event):
        if self.dragging_annotation is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        new_x = event.xdata - self.drag_offset[0]
        new_y = event.ydata - self.drag_offset[1]

        self.dragging_annotation.set_position((new_x, new_y))
        self.canvas.draw_idle()

    def on_release(self, event):
        self.dragging_annotation = None
        self.drag_offset = (0, 0)

    def clear_plot_elements(self):
        for marker in self.markers:
            try:
                marker.remove()
            except:
                pass
        for annotation in self.annotations:
            try:
                annotation.remove()
            except:
                pass
        for line in self.delta_lines:
            try:
                line.remove()
            except:
                pass

        self.markers.clear()
        self.annotations.clear()
        self.delta_lines.clear()
        self.selected_points.clear()
        self.dragging_annotation = None

    def normalized_signal(self):
        if self.df is None or self.y_columns_list.count() == 0:
            return
        try:
            selected_y_columns = []
            for i in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_y_columns.append(item.text())

            if not selected_y_columns:
                return
            self.df_normalized = self.df.copy()
            x_column = self.x_axis_combo.currentText()
            for column in selected_y_columns:
                if column != x_column:
                    data = self.df_normalized[column]
                    max_val = data.max()
                    min_val = data.min()
                    range_val = max_val - min_val
                    if range_val != 0:
                        self.df_normalized[column] = 2 * ((data - min_val) / range_val) - 1
                    else:
                        self.df_normalized[column] = 0
            self.is_normalized = True
            self.plot_signals()
        except Exception as e:
            print(f"Error normalizing signals: {e}")

    def update_filter_parameters(self):
        """Update visible filter parameters based on selected filter type"""
        for i in reversed(range(self.filter_params_layout.count())):
            widget = self.filter_params_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        filter_type = self.filter_type.currentText()

        if filter_type == "Low-pass":
            self.add_filter_param("Cutoff Frequency:", "100", "Hz", 0)

        elif filter_type == "High-pass":
            self.add_filter_param("Cutoff Frequency:", "10", "Hz", 0)

        elif filter_type == "Band-pass":
            self.add_filter_param("Low Frequency:", "20", "Hz", 0)
            self.add_filter_param("High Frequency:", "100", "Hz", 1)

        elif filter_type == "Band-stop":
            self.add_filter_param("Low Frequency:", "45", "Hz", 0)
            self.add_filter_param("High Frequency:", "55", "Hz", 1)

        elif filter_type == "Notch":
            self.add_filter_param("Notch Frequency:", "50", "Hz", 0)
            self.add_filter_param("Q Factor:", "30", "", 1)

    def add_filter_param(self, label_text, default_value, unit, row):
        """Helper method to add filter parameter widgets"""
        label = QLabel(label_text)
        input_field = QLineEdit(default_value)
        unit_label = QLabel(unit) if unit else QLabel("")

        self.filter_params_layout.addWidget(label, row, 0)
        self.filter_params_layout.addWidget(input_field, row, 1)
        self.filter_params_layout.addWidget(unit_label, row, 2)

    def apply_filter(self):
        """Apply selected filter to signals"""
        if self.df is None:
            return

        try:
            self.sampling_rate = float(self.sampling_rate_input.text())
            order = int(self.filter_order.text())
            filter_type = self.filter_type.currentText()
            if filter_type == "None":
                return
            if self.is_normalized and self.df_normalized is not None:
                data_to_filter = self.df_normalized
            else:
                data_to_filter = self.df
            self.df_filtered = data_to_filter.copy()
            x_column = self.x_axis_combo.currentText()
            selected_y_columns = []
            for i in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_y_columns.append(item.text())
            if not selected_y_columns:
                return
            for column in selected_y_columns:
                if column != x_column:
                    signal_data = data_to_filter[column].values
                    filtered_data = self.apply_specific_filter(signal_data, filter_type, order)
                    if filtered_data is not None:
                        self.df_filtered[column] = filtered_data
            self.is_filtered = True
            self.plot_signals()

        except Exception as e:
            print(f"Error applying filter: {e}")

    def apply_specific_filter(self, signal_data, filter_type, order):
        """Apply specific filter type to signal data"""
        try:
            nyquist = 0.5 * self.sampling_rate

            if filter_type == "Low-pass":
                cutoff = float(self.get_filter_param_value(0))
                normal_cutoff = cutoff / nyquist
                b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
                return signal.filtfilt(b, a, signal_data)
            elif filter_type == "High-pass":
                cutoff = float(self.get_filter_param_value(0))
                normal_cutoff = cutoff / nyquist
                b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
                return signal.filtfilt(b, a, signal_data)

            elif filter_type == "Band-pass":
                low_cutoff = float(self.get_filter_param_value(0))
                high_cutoff = float(self.get_filter_param_value(1))
                low_normal = low_cutoff / nyquist
                high_normal = high_cutoff / nyquist
                b, a = signal.butter(order, [low_normal, high_normal], btype='band', analog=False)
                return signal.filtfilt(b, a, signal_data)

            elif filter_type == "Band-stop":
                low_cutoff = float(self.get_filter_param_value(0))
                high_cutoff = float(self.get_filter_param_value(1))
                low_normal = low_cutoff / nyquist
                high_normal = high_cutoff / nyquist
                b, a = signal.butter(order, [low_normal, high_normal], btype='bandstop', analog=False)
                return signal.filtfilt(b, a, signal_data)

            elif filter_type == "Notch":
                notch_freq = float(self.get_filter_param_value(0))
                q_factor = float(self.get_filter_param_value(1))
                w0 = notch_freq / nyquist
                b, a = signal.iirnotch(w0, q_factor)
                return signal.filtfilt(b, a, signal_data)

            return None

        except Exception as e:
            print(f"Error in {filter_type} filter: {e}")
            return None

    def get_filter_param_value(self, index):
        """Get value from filter parameter field by index"""
        if self.filter_params_layout.count() > index * 3 + 1:
            input_field = self.filter_params_layout.itemAt(index * 3 + 1).widget()
            if isinstance(input_field, QLineEdit):
                return input_field.text()
        return "0"

    def remove_filter(self):
        """Remove applied filters"""
        self.is_filtered = False
        self.df_filtered = None
        self.plot_signals()

    def calculate_fft(self):
        """
        Calculate and display FFT of selected signals with configurable parameters
        """
        if self.df is None or self.y_columns_list.count() == 0:
            return

        try:
            # Clear previous FFT plot
            self.fft_figure.clear()
            self.fft_ax = self.fft_figure.add_subplot(111)
            self.fft_annotations.clear()
            self.fft_markers.clear()
            self.fft_selected_points.clear()

            # Get selected signals
            selected_y_columns = []
            for i in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_y_columns.append(item.text())

            if not selected_y_columns:
                return

            # Get data source
            if self.is_filtered and self.df_filtered is not None:
                data_source = self.df_filtered
            elif self.is_normalized and self.df_normalized is not None:
                data_source = self.df_normalized
            else:
                data_source = self.df

            x_column = self.x_axis_combo.currentText()
            time_data = data_source[x_column].values

            # Get FFT parameters
            sampling_rate = float(self.fft_sampling_rate.text())
            window_type = self.window_type.currentText()
            zero_padding = self.zero_padding.currentText()
            smoothing_type = self.smoothing_type.currentText()
            freq_min = float(self.freq_min.text())
            freq_max = float(self.freq_max.text())
            y_scale = self.y_scale.currentText()
            normalization = self.fft_normalization.currentText()

            # Calculate sampling rate from time data if available
            if len(time_data) > 1:
                actual_sampling_rate = 1.0 / (time_data[1] - time_data[0])
            else:
                actual_sampling_rate = sampling_rate

            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_y_columns)))
            legend_handles = []

            for i, y_column in enumerate(selected_y_columns):
                signal_data = data_source[y_column].values
                color = self.signal_colors.get(y_column, colors[i])

                # Apply selected window
                n = len(signal_data)
                if window_type == "Hanning":
                    window = np.hanning(n)
                elif window_type == "Hamming":
                    window = np.hamming(n)
                elif window_type == "Blackman":
                    window = np.blackman(n)
                elif window_type == "Bartlett":
                    window = np.bartlett(n)
                elif window_type == "Kaiser":
                    beta = float(self.kaiser_beta_input.text())
                    window = np.kaiser(n, beta)
                else:  # Rectangular
                    window = np.ones(n)

                windowed_signal = signal_data * window

                # Apply zero padding if selected
                if zero_padding == "None":
                    n_fft = n
                elif zero_padding == "2x":
                    n_fft = 2 * n
                elif zero_padding == "4x":
                    n_fft = 4 * n
                elif zero_padding == "8x":
                    n_fft = 8 * n
                elif zero_padding == "Custom":
                    n_fft = int(self.custom_padding_input.text())

                # Calculate FFT
                yf = fft(windowed_signal, n=n_fft)
                xf = fftfreq(n_fft, 1 / actual_sampling_rate)

                # Take only positive frequencies
                positive_freq_mask = (xf >= 0) & (xf <= freq_max)
                xf_positive = xf[positive_freq_mask]
                yf_positive = np.abs(yf[positive_freq_mask])

                # Apply normalization
                if normalization == "Amplitude":
                    yf_positive = 2.0 / n * yf_positive
                elif normalization == "Power":
                    yf_positive = (2.0 / n * yf_positive) ** 2
                elif normalization == "PSD":
                    yf_positive = (2.0 / n * yf_positive) ** 2 / (actual_sampling_rate / n)

                # Apply smoothing if selected
                if smoothing_type != "None":
                    window_size = int(self.smoothing_window_input.text())
                    if window_size % 2 == 0:  # Ensure window size is odd
                        window_size += 1

                    if smoothing_type == "Moving Average":
                        yf_smoothed = np.convolve(yf_positive, np.ones(window_size) / window_size, mode='same')
                    elif smoothing_type == "Savitzky-Golay":
                        yf_smoothed = signal.savgol_filter(yf_positive, window_size, 3)  # 3rd order polynomial
                else:
                    yf_smoothed = yf_positive

                # Apply frequency range filter
                freq_mask = (xf_positive >= freq_min) & (xf_positive <= freq_max)
                xf_filtered = xf_positive[freq_mask]
                yf_filtered = yf_smoothed[freq_mask]

                # Plot FFT
                line, = self.fft_ax.plot(xf_filtered, yf_filtered,
                                         label=y_column, color=color,
                                         linewidth=float(self.line_input.text()),
                                         alpha=0.8)
                legend_handles.append(line)

            # Set FFT plot properties
            self.fft_ax.set_xlabel('Frequency (Hz)', fontsize=20)

            # Set Y-axis label based on normalization
            if normalization == "Amplitude":
                ylabel = 'Amplitude'
            elif normalization == "Power":
                ylabel = 'Power'
            elif normalization == "PSD":
                ylabel = 'Power Spectral Density'
            else:
                ylabel = 'Magnitude'

            self.fft_ax.set_ylabel(ylabel, fontsize=20)

            # Set title with FFT parameters
            title = f'FFT Analysis - {self.title_input.text()}'
            if zero_padding != "None":
                title += f', {zero_padding} Zero Padding'
            if smoothing_type != "None":
                title += f', {smoothing_type} Smoothing'
            self.fft_ax.set_title(title, fontsize=16, fontweight='bold')

            self.fft_ax.grid(False)
            self.fft_ax.legend(handles=legend_handles, loc='best', framealpha=0.9, fontsize=18)

            # Set Y-axis scale
            if y_scale == "Logarithmic":
                self.fft_ax.set_yscale('log')

            # Set X-axis limits
            self.fft_ax.set_xlim(freq_min, freq_max)

            # Remove top and right spines for cleaner look
            self.fft_ax.spines['top'].set_visible(False)
            self.fft_ax.spines['right'].set_visible(False)

            self.fft_figure.tight_layout()
            self.fft_canvas.draw_idle()

            # Switch to FFT tab
            self.plot_tabs.setCurrentIndex(1)

        except Exception as e:
            print(f"Error calculating FFT: {e}")

    def on_fft_click(self, event):
        """
        Handle clicks on FFT plot for point selection
        Right click to clear all selections
        Left click to add multiple points with non-overlapping annotations
        Middle click or drag to move annotations
        """
        if event.inaxes != self.fft_ax:
            return

            # Right click to clear all selections
        if event.button == 3:
            self.clear_fft_selections()
            self.fft_canvas.draw_idle()
            return

            # Check if clicking on existing annotation to drag it
        if event.button == 1:
            for i, annotation in enumerate(self.fft_annotations):
                contains, _ = annotation.contains(event)
                if contains:
                    self.dragging_fft_annotation = annotation
                    self.dragging_fft_index = i
                    # Store the current mouse position
                    self.drag_start_x = event.x
                    self.drag_start_y = event.y
                    # Store the current annotation offset
                    self.drag_current_offset = annotation.xyann
                    return

            # Left click to add points (only if not dragging and not on toolbar)
        if event.button != 1 or self.fft_toolbar.mode != '':
            return

        x_click = event.xdata
        y_click = event.ydata

        if x_click is None or y_click is None or x_click < 0 or x_click > 25:
            return

        # Find the closest point on each signal
        closest_points = []
        for i, line in enumerate(self.fft_ax.get_lines()):
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Find closest point within the visible range
            visible_mask = (x_data >= 0) & (x_data <= 25)
            x_visible = x_data[visible_mask]
            y_visible = y_data[visible_mask]

            if len(x_visible) > 0:
                idx = np.abs(x_visible - x_click).argmin()
                x_point = x_visible[idx]
                y_point = y_visible[idx]

                # Calculate distance (log scale for y)
                dist = np.hypot(x_click - x_point, np.log10(max(y_click, 1e-10)) - np.log10(max(y_point, 1e-10)))
                closest_points.append((dist, x_point, y_point, line, i))

        if closest_points:
            # Find the closest point among all signals
            closest_points.sort(key=lambda x: x[0])
            dist, x_point, y_point, line, line_idx = closest_points[0]

            if dist < 0.2:  # Adjust threshold as needed
                signal_name = line.get_label()

                # Check if this point is already selected
                for existing_point in self.fft_selected_points:
                    if (abs(existing_point['x'] - x_point) < 0.01 and
                            abs(existing_point['y'] - y_point) < 0.01 and
                            existing_point['signal'] == signal_name):
                        return  # Point already exists, don't add duplicate

                # Create marker with unique color for each signal
                color = line.get_color()
                marker, = self.fft_ax.plot(x_point, y_point, 'o', markersize=10, alpha=0.9,
                                           zorder=5, color=color, markeredgecolor='black',
                                           markeredgewidth=1.5)
                self.fft_markers.append(marker)

                # Create annotation with non-overlapping positioning and larger text box
                annotation_text = f"{signal_name}\nFreq: {x_point:.3f} Hz\nAmp: {y_point:.6f}"

                # Calculate optimal position to avoid overlaps
                xytext = self.calculate_annotation_position(x_point, y_point)

                annotation = self.fft_ax.annotate(
                    annotation_text,
                    xy=(x_point, y_point),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        fc="lightyellow",
                        alpha=0.95,
                        ec="black",
                        linewidth=1.5
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        color=color,
                        alpha=0.8,
                        linewidth=2.0
                    )
                )
                self.fft_annotations.append(annotation)

                # Store point information
                self.fft_selected_points.append({
                    'x': x_point,
                    'y': y_point,
                    'signal': signal_name,
                    'marker': marker,
                    'annotation': annotation,
                    'position': xytext,
                    'draggable': True
                })

                self.fft_canvas.draw_idle()

    def on_fft_motion(self, event):
        """
        Handle mouse motion for dragging FFT annotations
        """
        if (self.dragging_fft_annotation is None or event.inaxes != self.fft_ax):
            return

        # Calculate the drag distance in pixels
        dx_pixels = event.x - self.drag_start_x
        dy_pixels = event.y - self.drag_start_y

        # Convert pixel movement to offset points (approximate conversion)
        # A reasonable scaling factor: 1 pixel ≈ 0.75 points
        dx_points = dx_pixels * 0.75
        dy_points = dy_pixels * -0.75  # Negative because y-axis is inverted in display coordinates

        # Update annotation position
        new_dx = self.drag_current_offset[0] + dx_points
        new_dy = self.drag_current_offset[1] + dy_points

        self.dragging_fft_annotation.xyann = (new_dx, new_dy)

        self.fft_canvas.draw_idle()

    def on_fft_release(self, event):
        """
        Handle mouse release after dragging FFT annotations
        """
        if self.dragging_fft_annotation is not None and self.dragging_fft_index is not None:
            # Update the stored position with the final offset
            final_offset = self.dragging_fft_annotation.xyann
            self.fft_selected_points[self.dragging_fft_index]['position'] = final_offset

        self.dragging_fft_annotation = None
        self.dragging_fft_index = None
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_current_offset = None

    def calculate_annotation_position(self, x, y):
        """
        Calculate optimal annotation position to avoid overlaps
        Uses a spiral pattern to find the best position with larger bounding boxes
        """
        # Initial positions to try (right, left, top, bottom, diagonals)
        positions = [
            (25, 25),  # Right-bottom
            (25, -35),  # Right-top
            (-100, 25),  # Left-bottom
            (-100, -35),  # Left-top
            (0, 50),  # Top
            (0, -60),  # Bottom
            (50, 0),  # Right
            (-110, 0)  # Left
        ]

        # If no existing annotations, use the first position
        if not self.fft_selected_points:
            return positions[0]

        # Check each position for overlaps with larger bounding box
        for dx, dy in positions:
            overlap = False
            new_bbox = (x + dx / 80, y + dy / 80, 0.20, 0.12)  # Larger bbox size

            for existing_point in self.fft_selected_points:
                if 'position' in existing_point:
                    ex, ey = existing_point['x'], existing_point['y']
                    edx, edy = existing_point['position']
                    existing_bbox = (ex + edx / 80, ey + edy / 80, 0.20, 0.12)

                    # Check if bounding boxes overlap
                    if self.bbox_overlap(new_bbox, existing_bbox):
                        overlap = True
                        break

            if not overlap:
                return (dx, dy)

        # If all positions overlap, use a spiral pattern with larger radius
        max_attempts = 16
        for attempt in range(1, max_attempts + 1):
            # Spiral outwards with larger steps
            angle = attempt * (2 * np.pi / 8)
            radius = 25 + attempt * 8  # Larger radius steps
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)

            overlap = False
            new_bbox = (x + dx / 80, y + dy / 80, 0.20, 0.12)  # Larger bbox

            for existing_point in self.fft_selected_points:
                if 'position' in existing_point:
                    ex, ey = existing_point['x'], existing_point['y']
                    edx, edy = existing_point['position']
                    existing_bbox = (ex + edx / 80, ey + edy / 80, 0.20, 0.12)

                    if self.bbox_overlap(new_bbox, existing_bbox):
                        overlap = True
                        break

            if not overlap:
                return (dx, dy)

        # Fallback to first position if all else fails
        return positions[0]

    def bbox_overlap(self, bbox1, bbox2):
        """
        Check if two bounding boxes overlap
        bbox format: (center_x, center_y, width, height)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Check if bounding boxes overlap with larger tolerance
        return (abs(x1 - x2) * 2 < (w1 + w2)) and (abs(y1 - y2) * 2 < (h1 + h2))

    def clear_fft_selections(self):
        """Clear all FFT selections (markers and annotations)"""
        for marker in self.fft_markers:
            try:
                marker.remove()
            except:
                pass
        for annotation in self.fft_annotations:
            try:
                annotation.remove()
            except:
                pass

        self.fft_markers.clear()
        self.fft_annotations.clear()
        self.fft_selected_points.clear()
        self.dragging_fft_annotation = None
        self.dragging_fft_index = None
        self.drag_fft_offset = (0, 0)

    def stft(self):
        """Placeholder for STFT functionality"""
        pass

    def calculate_cwt(self):
        """
        Calculate and display Continuous Wavelet Transform of selected signals
        with respect to the assigned X-axis range
        """
        if self.df is None or self.y_columns_list.count() == 0:
            return

        try:
            # Clear previous CWT plot
            self.cwt_figure.clear()
            self.cwt_ax = self.cwt_figure.add_subplot(111)
            self.cwt_annotations.clear()
            self.cwt_markers.clear()

            # Get selected signals
            selected_y_columns = []
            for i in range(self.y_columns_list.count()):
                item = self.y_columns_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_y_columns.append(item.text())

            if not selected_y_columns:
                return

            # Get data source
            if self.is_filtered and self.df_filtered is not None:
                data_source = self.df_filtered
            elif self.is_normalized and self.df_normalized is not None:
                data_source = self.df_normalized
            else:
                data_source = self.df

            x_column = self.x_axis_combo.currentText()

            # Apply X-axis range filtering
            x_min = self.x_min_input.text().strip()
            x_max = self.x_max_input.text().strip()
            filtered_data = data_source.copy()

            if x_min:
                try:
                    x_min_val = float(x_min)
                    filtered_data = filtered_data[filtered_data[x_column] >= x_min_val]
                except ValueError:
                    pass
            if x_max:
                try:
                    x_max_val = float(x_max)
                    filtered_data = filtered_data[filtered_data[x_column] <= x_max_val]
                except ValueError:
                    pass

            time_data = filtered_data[x_column].values

            # Get CWT parameters
            wavelet = self.wavelet_type.currentText()
            scales_min = int(self.scales_min.text())
            scales_max = int(self.scales_max.text())
            num_scales = int(self.num_scales.text())
            scales = np.linspace(scales_min, scales_max, num_scales)

            # Get sampling rate
            sampling_rate = float(self.cwt_sampling_rate.text())
            if len(time_data) > 1:
                actual_sampling_rate = 1.0 / (time_data[1] - time_data[0])
            else:
                actual_sampling_rate = sampling_rate

            # Calculate frequencies corresponding to scales
            center_freq = pywt.central_frequency(wavelet)
            frequencies = center_freq * actual_sampling_rate / scales

            # Plot each selected signal
            for y_column in selected_y_columns:
                signal_data = filtered_data[y_column].values

                # Perform CWT only if we have data in the selected range
                if len(signal_data) > 0:
                    coefficients, freqs = pywt.cwt(signal_data, scales, wavelet,
                                                   sampling_period=1.0 / actual_sampling_rate)

                    # Plot CWT with the actual time range
                    im = self.cwt_ax.imshow(np.abs(coefficients),
                                            extent=[time_data[0], time_data[-1],
                                                    frequencies[-1], frequencies[0]],
                                            aspect='auto',
                                            cmap=self.colormap.currentText(),
                                            vmin=float(self.cwt_min.text()),
                                            vmax=float(self.cwt_max.text()))

                    # Add colorbar
                    self.cwt_figure.colorbar(im, ax=self.cwt_ax, label='Magnitude')

            # Set CWT plot properties
            self.cwt_ax.set_xlabel('Time (s)', fontsize=20)
            self.cwt_ax.set_ylabel('Frequency (Hz)', fontsize=20)
            self.cwt_ax.set_title(f'CWT Analysis - {wavelet} Wavelet', fontsize=16, fontweight='bold')
            self.cwt_ax.grid(False)

            # Set logarithmic scale for frequency axis if needed
            if scales_max / scales_min > 10:
                self.cwt_ax.set_yscale('log')

            self.cwt_figure.tight_layout()
            self.cwt_canvas.draw_idle()

            # Switch to CWT tab
            self.plot_tabs.setCurrentIndex(2)

        except Exception as e:
            print(f"Error calculating CWT: {e}")

    def wvd(self):
        """Placeholder for WVD functionality"""
        pass


def main():
    app = QApplication(sys.argv)
    style_sheet = """
            /* Estilos generales para los 4 paneles para hacerlos visibles */
            #top_panel {
                background-color: #F8F8F8; /* GhostWhite */
                border-radius: 5px;
            }
            #bottom_panel {
                background-color: #F0F8FF; /* AliceBlue */
                border-radius: 5px;
            }
            #left_panel {
                background-color: #F5F5F5; /* WhiteSmoke */
                border-radius: 5px;
            }
            #right_panel {
                background-color: #FFFFFF; /* White */
                border-radius: 5px;
            }

            /* Estilo base para todos los botones */
            QPushButton {
                color: white;
                padding: 6px;
                border-radius: 5px;
                font-weight: bold;
                border: 1px solid rgba(0, 0, 0, 0.2);
            }

            /* 🟢 Botones HABILITADOS (verdes) */
            QPushButton:enabled {
                background-color: #2ecc71; /* verde */
            }
            QPushButton:enabled:hover {
                background-color: #27ae60; /* verde más oscuro */
            }

            /* 🔴 Botones INHABILITADOS (rojos) */
            QPushButton:disabled {
                background-color: #e74c3c; /* rojo */
                color: #ecf0f1;
            }

            /* 🔵 Botón de calibración específico (azul) - SOBREESCRIBE los demás */
            QPushButton#calibrate_btn {
                background-color: #3498db; /* azul */
            }
            QPushButton#calibrate_btn:hover {
                background-color: #2980b9; /* azul más oscuro */
            }

            /* Estilos para otros widgets */
            QLineEdit, QComboBox {
                padding: 3px;
                border: 1px solid gray;
                border-radius: 4px;
            }
        """
    app.setStyleSheet(style_sheet)
    window = SignalAnalyzer()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()