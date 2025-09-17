import sys
import pandas as pd
import numpy as np
from scipy import signal
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
        self.sampling_rate = 1000
        self.setup()

    def setup(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Top Panel ---
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

        for btn in [self.load_btn, self.plot_btn, self.clear_btn, self.normalized_btn]:
            btn.setStyleSheet(button_style)

        header_layout.addWidget(self.load_btn)
        header_layout.addWidget(self.plot_btn)
        header_layout.addWidget(self.clear_btn)
        header_layout.addWidget(self.legend_btn)
        header_layout.addWidget(self.normalized_btn)
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
        Tab 2: Filters Controls - CON SCROLL AREA
        """
        filters_tab = QWidget()
        filters_layout = QVBoxLayout(filters_tab)
        """
        Create scroll area for filters
        """
        filters_scroll = QScrollArea()
        filters_scroll.setWidgetResizable(True)
        filters_scroll.setMaximumHeight(300)  # Altura máxima para el scroll
        """
        Container widget for scroll area
        """
        filters_container = QWidget()
        filters_container_layout = QVBoxLayout(filters_container)

        filters_group = QGroupBox("Filter Settings")
        self.filters_grid = QGridLayout(filters_group)

        """
        Sampling rate
        """
        self.filters_grid.addWidget(QLabel("Sampling Rate:"), 0, 0)
        self.sampling_rate_input = QLineEdit('100000')
        self.filters_grid.addWidget(self.sampling_rate_input, 0, 1)
        self.filters_grid.addWidget(QLabel("Hz"), 0, 2)
        """
        Filter type
        """
        self.filters_grid.addWidget(QLabel("Filter Type:"), 1, 0)
        self.filter_type = QComboBox()
        self.filter_type.addItems(["None", "Low-pass", "High-pass", "Band-pass", "Notch", "Band-stop"])
        self.filter_type.currentTextChanged.connect(self.update_filter_parameters)
        self.filters_grid.addWidget(self.filter_type, 1, 1, 1, 2)
        """
        Filter order
        """
        self.filters_grid.addWidget(QLabel("Filter Order:"), 2, 0)
        self.filter_order = QLineEdit('4')
        self.filters_grid.addWidget(self.filter_order, 2, 1, 1, 2)

        """
        Filter parameters container
        """
        self.filter_params_container = QWidget()
        self.filter_params_layout = QGridLayout(self.filter_params_container)
        self.filters_grid.addWidget(self.filter_params_container, 3, 0, 1, 3)

        """
        Apply and Remove buttons
        """
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.filters_grid.addWidget(self.apply_filter_btn, 4, 0, 1, 3)

        self.remove_filter_btn = QPushButton("Remove Filter")
        self.filters_grid.addWidget(self.remove_filter_btn, 5, 0, 1, 3)

        """
        Add filter group to container
        """
        filters_container_layout.addWidget(filters_group)
        filters_container_layout.addStretch()

        """
        Set the container as scroll area widget
        """
        filters_scroll.setWidget(filters_container)

        """
        Add scroll area to filters layout
        """
        filters_layout.addWidget(filters_scroll)

        controls_tabs.addTab(graph_tab, "Graph Controls")
        controls_tabs.addTab(filters_tab, "Filters Controls")
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
        Matplotlib Figure
        """
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

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
        self.x_axis_combo.currentTextChanged.connect(self.update_y_columns_list)

        """
        Event Connectors
        """
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        """
        Initialize filter parameters
        """
        self.update_filter_parameters()

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

    def fft(self):
        pass

    def stft(self):
        pass

    def cwt(self):
        pass

    def wvd(self):
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