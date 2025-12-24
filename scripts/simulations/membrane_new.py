import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QDoubleSpinBox,
                               QSlider, QGridLayout, QTextEdit, QSplitter,
                               QComboBox, QCheckBox, QPushButton)
from PySide6.QtCore import Qt, Signal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('Qt5Agg')


class MylarHydrophoneSimulator(QMainWindow):
    """
    Realistic optical hydrophone simulator for mylar membranes.
    Uses correct physics: combined tension + bending stiffness model.
    """
    parametersChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mylar Optical Hydrophone Simulator")
        self.setGeometry(100, 100, 1600, 950)

        # Debounce timer for slider updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)  # 150ms debounce
        self.update_timer.timeout.connect(self.update_all)

        # Initialize parameters
        self.init_parameters()
        self.init_ui()
        self.parametersChanged.connect(self.schedule_update)

    def init_parameters(self):
        """Initialize parameters with realistic mylar properties."""
        self.params = {
            # Acoustic input (in Pa)
            'pressure_pa': 1.0,

            # Mylar membrane properties (space blanket)
            'membrane_diameter_mm': 20.0,
            'membrane_thickness_um': 25.0,
            'membrane_tension_Nm': 1.0,  # Realistic: 0.1-10 N/m
            'reflectivity': 0.90,

            # Material properties for Mylar (PET)
            'youngs_modulus_GPa': 3.5,  # 2.8-4.5 GPa typical
            'poissons_ratio': 0.38,  # Mylar specific

            # Laser position
            'radial_offset_percent': 50.0,

            # Optical autocollimator system
            'focal_length_m': 0.5,
            'pixel_size_um': 4.86,
            'camera_noise_pixels': 0.5,

            # Display settings
            'pressure_unit': 'Pa',
            'use_log_scale': True,
        }

        self.unit_conversion = {
            'μPa': 1e-6,
            'Pa': 1.0,
            'kPa': 1e3,
            'MPa': 1e6
        }

        self.results = {}

    def init_ui(self):
        """Initialize the UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right Panel: Visualization and Results
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Matplotlib plots
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        self.canvas = FigureCanvas(self.figure)
        right_splitter.addWidget(self.canvas)

        # Bottom: Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setMinimumHeight(200)
        right_splitter.addWidget(self.results_display)
        right_splitter.setSizes([600, 250])

        main_layout.addWidget(right_splitter, stretch=3)

        # Initial calculation
        self.update_all()

    def create_control_panel(self):
        """Create control panel with realistic ranges."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title
        title = QLabel("<h2>Mylar Hydrophone Simulator</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)

        # === 1. Acoustic Pressure ===
        pressure_group = QGroupBox("1. Acoustic Pressure")
        pressure_layout = QGridLayout()

        # Unit selector
        pressure_layout.addWidget(QLabel("Unit:"), 0, 0)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['μPa', 'Pa', 'kPa', 'MPa'])
        self.unit_combo.setCurrentText('Pa')
        self.unit_combo.currentTextChanged.connect(self.on_unit_change)
        pressure_layout.addWidget(self.unit_combo, 0, 1)

        # Log scale checkbox
        self.log_checkbox = QCheckBox("Log scale")
        self.log_checkbox.setChecked(True)
        self.log_checkbox.stateChanged.connect(self.on_log_scale_change)
        pressure_layout.addWidget(self.log_checkbox, 0, 2)

        # Pressure spinbox
        pressure_layout.addWidget(QLabel("Value:"), 1, 0)
        self.pressure_spinbox = QDoubleSpinBox()
        self.pressure_spinbox.setDecimals(6)
        self.pressure_spinbox.setRange(1e-12, 1e12)
        self.pressure_spinbox.setValue(1.0)
        self.pressure_spinbox.valueChanged.connect(self.on_pressure_change)
        pressure_layout.addWidget(self.pressure_spinbox, 1, 1, 1, 2)

        # Log slider
        self.pressure_slider = QSlider(Qt.Orientation.Horizontal)
        self.pressure_slider.setRange(0, 1000)
        self.pressure_slider.setValue(self.pressure_to_slider(1.0))
        self.pressure_slider.valueChanged.connect(self.on_slider_change)
        pressure_layout.addWidget(self.pressure_slider, 2, 0, 1, 3)

        # Presets
        preset_layout = QHBoxLayout()
        presets = [
            ("Ocean\nAmbient\n50 μPa", 50e-6),
            ("Whale\nCall\n1 mPa", 1e-3),
            ("Ship\nNoise\n1 Pa", 1.0),
            ("Sonar\n10 kPa", 10e3)
        ]
        for label, value in presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, v=value: self.set_pressure_preset(v))
            preset_layout.addWidget(btn)
        pressure_layout.addLayout(preset_layout, 3, 0, 1, 3)

        pressure_group.setLayout(pressure_layout)
        layout.addWidget(pressure_group)

        # === 2. Mylar Membrane ===
        membrane_group = QGroupBox("2. Mylar Membrane")
        membrane_layout = QGridLayout()

        # Diameter
        self.dia_slider, self.dia_spinbox = self.create_param_control(
            "Diameter:", 5.0, 100.0, 20.0, "mm", membrane_layout, 0
        )

        # Thickness
        self.thick_slider, self.thick_spinbox = self.create_param_control(
            "Thickness:", 10.0, 25.0, 15.0, "μm", membrane_layout, 1
        )

        # Tension with mounting presets
        membrane_layout.addWidget(QLabel("Mounting:"), 2, 0)
        self.mounting_combo = QComboBox()
        self.mounting_combo.addItems(['Slack', 'Light', 'Moderate', 'Tight', 'Custom'])
        self.mounting_combo.setCurrentText('Light')
        self.mounting_combo.currentTextChanged.connect(self.on_mounting_change)
        membrane_layout.addWidget(self.mounting_combo, 2, 1, 1, 2)

        # Tension control
        self.tens_slider, self.tens_spinbox = self.create_param_control(
            "Tension:", 0.1, 10.0, 1.0, "N/m", membrane_layout, 3
        )

        # Physics mode indicator
        self.physics_label = QLabel("Regime: Calculating...")
        self.physics_label.setStyleSheet("color: #666; font-size: 10px;")
        membrane_layout.addWidget(self.physics_label, 4, 0, 1, 3)

        membrane_group.setLayout(membrane_layout)
        layout.addWidget(membrane_group)

        # === 3. Optical System ===
        optical_group = QGroupBox("3. Autocollimator Setup")
        optical_layout = QGridLayout()

        # Laser offset
        self.offset_slider, self.offset_spinbox = self.create_param_control(
            "Laser Offset:", 10.0, 90.0, 50.0, "%", optical_layout, 0
        )

        # Focal length
        self.focal_slider, self.focal_spinbox = self.create_param_control(
            "Focal Length:", 0.1, 2.0, 0.5, "m", optical_layout, 1
        )

        # Pixel size
        optical_layout.addWidget(QLabel("Pixel Size:"), 2, 0)
        self.pixel_spinbox = QDoubleSpinBox()
        self.pixel_spinbox.setRange(1.0, 20.0)
        self.pixel_spinbox.setValue(4.86)
        self.pixel_spinbox.setSuffix(" μm")
        self.pixel_spinbox.valueChanged.connect(self.on_parameter_change)
        optical_layout.addWidget(self.pixel_spinbox, 2, 1, 1, 2)

        # Camera noise
        optical_layout.addWidget(QLabel("Noise (RMS):"), 3, 0)
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.01, 5.0)
        self.noise_spinbox.setValue(0.1)
        self.noise_spinbox.setSuffix(" px")
        self.noise_spinbox.setSingleStep(0.01)
        self.noise_spinbox.valueChanged.connect(self.on_parameter_change)
        optical_layout.addWidget(self.noise_spinbox, 3, 1, 1, 2)

        optical_group.setLayout(optical_layout)
        layout.addWidget(optical_group)

        # === 4. Material Properties ===
        material_group = QGroupBox("4. Material Properties")
        material_layout = QGridLayout()

        material_layout.addWidget(QLabel("Young's Modulus:"), 0, 0)
        self.youngs_spinbox = QDoubleSpinBox()
        self.youngs_spinbox.setRange(2.0, 6.0)
        self.youngs_spinbox.setValue(3.5)
        self.youngs_spinbox.setSuffix(" GPa")
        self.youngs_spinbox.setSingleStep(0.1)
        self.youngs_spinbox.valueChanged.connect(self.on_parameter_change)
        material_layout.addWidget(self.youngs_spinbox, 0, 1)

        material_layout.addWidget(QLabel("Poisson's Ratio:"), 1, 0)
        self.poisson_spinbox = QDoubleSpinBox()
        self.poisson_spinbox.setRange(0.2, 0.5)
        self.poisson_spinbox.setValue(0.38)
        self.poisson_spinbox.setSingleStep(0.01)
        self.poisson_spinbox.valueChanged.connect(self.on_parameter_change)
        material_layout.addWidget(self.poisson_spinbox, 1, 1)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        layout.addStretch(1)
        return panel

    def create_param_control(self, label, min_val, max_val, init_val, unit, layout, row):
        """Helper to create parameter control with slider and spinbox."""
        layout.addWidget(QLabel(label), row, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(int(((init_val - min_val) / (max_val - min_val)) * 1000))
        slider.valueChanged.connect(lambda: self.on_parameter_change())
        layout.addWidget(slider, row, 1)

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(init_val)
        spinbox.setSuffix(f" {unit}")
        spinbox.setSingleStep((max_val - min_val) / 100)
        spinbox.valueChanged.connect(lambda: self.on_parameter_change())
        layout.addWidget(spinbox, row, 2)

        # Link slider and spinbox
        slider.valueChanged.connect(
            lambda val: spinbox.setValue(min_val + (val / 1000.0) * (max_val - min_val))
        )
        spinbox.valueChanged.connect(
            lambda val: slider.setValue(int(((val - min_val) / (max_val - min_val)) * 1000))
        )

        return slider, spinbox

    def slider_to_pressure(self, slider_val):
        """Convert slider value to pressure (log scale)."""
        log_min, log_max = np.log10(1e-6), np.log10(1e8)
        log_val = log_min + (slider_val / 1000.0) * (log_max - log_min)
        return 10 ** log_val

    def pressure_to_slider(self, pressure_pa):
        """Convert pressure to slider value (log scale)."""
        log_min, log_max = np.log10(1e-6), np.log10(1e8)
        log_val = np.log10(max(pressure_pa, 1e-6))
        log_val = np.clip(log_val, log_min, log_max)
        return int(((log_val - log_min) / (log_max - log_min)) * 1000)

    def on_unit_change(self):
        """Handle pressure unit change."""
        unit = self.unit_combo.currentText()
        pressure_pa = self.params['pressure_pa']
        value = pressure_pa / self.unit_conversion[unit]

        self.pressure_spinbox.blockSignals(True)
        self.pressure_spinbox.setValue(value)
        self.pressure_spinbox.blockSignals(False)

        self.params['pressure_unit'] = unit

    def on_log_scale_change(self):
        """Toggle logarithmic scale."""
        self.params['use_log_scale'] = self.log_checkbox.isChecked()
        self.schedule_update()

    def on_pressure_change(self):
        """Handle pressure spinbox change."""
        unit = self.unit_combo.currentText()
        value = self.pressure_spinbox.value()
        self.params['pressure_pa'] = value * self.unit_conversion[unit]

        # Update slider
        slider_val = self.pressure_to_slider(self.params['pressure_pa'])
        self.pressure_slider.blockSignals(True)
        self.pressure_slider.setValue(slider_val)
        self.pressure_slider.blockSignals(False)

        self.schedule_update()

    def on_slider_change(self):
        """Handle pressure slider change."""
        pressure_pa = self.slider_to_pressure(self.pressure_slider.value())
        self.params['pressure_pa'] = pressure_pa

        # Update spinbox
        unit = self.unit_combo.currentText()
        value = pressure_pa / self.unit_conversion[unit]
        self.pressure_spinbox.blockSignals(True)
        self.pressure_spinbox.setValue(value)
        self.pressure_spinbox.blockSignals(False)

        self.schedule_update()

    def set_pressure_preset(self, pressure_pa):
        """Set pressure to preset value."""
        self.params['pressure_pa'] = pressure_pa

        unit = self.unit_combo.currentText()
        value = pressure_pa / self.unit_conversion[unit]
        self.pressure_spinbox.blockSignals(True)
        self.pressure_spinbox.setValue(value)
        self.pressure_spinbox.blockSignals(False)

        slider_val = self.pressure_to_slider(pressure_pa)
        self.pressure_slider.blockSignals(True)
        self.pressure_slider.setValue(slider_val)
        self.pressure_slider.blockSignals(False)

        self.schedule_update()

    def on_mounting_change(self):
        """Handle mounting preset change."""
        mounting = self.mounting_combo.currentText()
        tension_map = {
            'Slack': 0.2,
            'Light': 1.0,
            'Moderate': 3.0,
            'Tight': 7.0,
            'Custom': self.tens_spinbox.value()
        }

        if mounting != 'Custom':
            tension = tension_map[mounting]
            self.tens_spinbox.blockSignals(True)
            self.tens_spinbox.setValue(tension)
            self.tens_spinbox.blockSignals(False)
            self.on_parameter_change()

    def on_parameter_change(self):
        """Update parameters from UI."""
        self.params['membrane_diameter_mm'] = self.dia_spinbox.value()
        self.params['membrane_thickness_um'] = self.thick_spinbox.value()
        self.params['membrane_tension_Nm'] = self.tens_spinbox.value()
        self.params['radial_offset_percent'] = self.offset_spinbox.value()
        self.params['focal_length_m'] = self.focal_spinbox.value()
        self.params['pixel_size_um'] = self.pixel_spinbox.value()
        self.params['camera_noise_pixels'] = self.noise_spinbox.value()
        self.params['youngs_modulus_GPa'] = self.youngs_spinbox.value()
        self.params['poissons_ratio'] = self.poisson_spinbox.value()

        # Check if tension was manually changed
        if abs(self.tens_spinbox.value() - {
            'Slack': 0.2, 'Light': 1.0, 'Moderate': 3.0, 'Tight': 7.0
        }.get(self.mounting_combo.currentText(), -1)) > 0.01:
            self.mounting_combo.blockSignals(True)
            self.mounting_combo.setCurrentText('Custom')
            self.mounting_combo.blockSignals(False)

        self.schedule_update()

    def schedule_update(self):
        """Schedule update with debouncing."""
        self.update_timer.start()

    def calculate_physics(self):
        """Calculate membrane response with CORRECT physics."""
        P = self.params['pressure_pa']
        R = (self.params['membrane_diameter_mm'] / 1000.0) / 2.0
        h = self.params['membrane_thickness_um'] * 1e-6
        T = self.params['membrane_tension_Nm']
        E = self.params['youngs_modulus_GPa'] * 1e9
        nu = self.params['poissons_ratio']
        r_offset = (self.params['radial_offset_percent'] / 100.0) * R

        # === CORRECTED MEMBRANE DEFLECTION ===
        # Bending rigidity: D = Eh³/[12(1-ν²)]
        D = (E * h ** 3) / (12 * (1 - nu ** 2))

        # Effective stiffness contribution from bending
        K_bending = 4 * D / (R ** 2)

        # Total effective tension: T_eff = T + K_bending
        T_eff = T + K_bending

        # Center deflection with combined model
        if T_eff > 0:
            delta_center = (P * R ** 2) / (4 * T_eff)
        else:
            delta_center = 0.0

        # Determine dominant regime
        tension_contrib = T / T_eff if T_eff > 0 else 0
        bending_contrib = K_bending / T_eff if T_eff > 0 else 0

        if tension_contrib > 0.9:
            regime = "Tension-dominated"
        elif bending_contrib > 0.9:
            regime = "Bending-dominated"
        else:
            regime = f"Mixed (T:{tension_contrib * 100:.0f}% B:{bending_contrib * 100:.0f}%)"

        # Store results
        self.results['center_deflection_m'] = delta_center
        self.results['center_deflection_nm'] = delta_center * 1e9
        self.results['bending_rigidity'] = D
        self.results['effective_tension'] = T_eff
        self.results['regime'] = regime
        self.results['tension_contribution'] = tension_contrib
        self.results['bending_contribution'] = bending_contrib

        # === SLOPE AT LASER POINT ===
        # For parabolic profile: w(r) = w₀(1 - r²/R²)
        # Slope: dw/dr = -2w₀r/R²
        if R > 0 and delta_center != 0:
            slope_rad = abs((2.0 * delta_center * r_offset) / (R ** 2))
        else:
            slope_rad = 0.0

        self.results['slope_angle_rad'] = slope_rad
        self.results['slope_angle_urad'] = slope_rad * 1e6

        # === AUTOCOLLIMATOR RESPONSE ===
        # Reflected beam angle: 2θ (autocollimator principle)
        reflected_angle = 2.0 * slope_rad

        # Spot displacement on detector
        f = self.params['focal_length_m']
        spot_displacement_m = f * reflected_angle

        self.results['reflected_angle_rad'] = reflected_angle
        self.results['spot_displacement_m'] = spot_displacement_m
        self.results['spot_displacement_um'] = spot_displacement_m * 1e6
        self.results['spot_displacement_nm'] = spot_displacement_m * 1e9

        # === PIXEL DISPLACEMENT ===
        pix_size = self.params['pixel_size_um'] * 1e-6
        if pix_size > 0:
            displacement_pixels = spot_displacement_m / pix_size
        else:
            displacement_pixels = 0.0

        self.results['displacement_pixels'] = displacement_pixels

        # === SNR CALCULATION ===
        noise = self.params['camera_noise_pixels']
        if noise > 0:
            snr_linear = displacement_pixels / noise
            snr_db = 20 * np.log10(max(snr_linear, 1e-12))
        else:
            snr_linear = float('inf') if displacement_pixels > 0 else 0.0
            snr_db = float('inf') if displacement_pixels > 0 else -float('inf')

        self.results['snr_linear'] = snr_linear
        self.results['snr_db'] = snr_db

        # === SENSITIVITY ===
        if P > 0:
            sensitivity_px_per_pa = displacement_pixels / P
            sensitivity_nm_per_pa = delta_center * 1e9 / P
        else:
            sensitivity_px_per_pa = 0.0
            sensitivity_nm_per_pa = 0.0

        self.results['sensitivity_px_per_pa'] = sensitivity_px_per_pa
        self.results['sensitivity_nm_per_pa'] = sensitivity_nm_per_pa

        # === MINIMUM DETECTABLE PRESSURE ===
        # Assuming 0.1 pixel detection threshold
        detection_threshold_px = 0.1
        if sensitivity_px_per_pa > 0:
            min_detectable_pa = detection_threshold_px / sensitivity_px_per_pa
        else:
            min_detectable_pa = float('inf')

        self.results['min_detectable_pa'] = min_detectable_pa
        self.results['min_detectable_upa'] = min_detectable_pa * 1e6

        # Store geometry
        self.results['pressure_pa'] = P
        self.results['radius_m'] = R
        self.results['laser_offset_m'] = r_offset

        # Update physics label
        self.physics_label.setText(f"Regime: {regime}")

    def update_all(self):
        """Update calculations and visualizations."""
        self.calculate_physics()
        self.update_plots()
        self.update_results_text()

    def update_plots(self):
        """Update all plots efficiently."""
        self.figure.clear()

        # Create subplot grid
        gs = self.figure.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # Plot 1: System schematic
        ax1 = self.figure.add_subplot(gs[0, :])
        self.plot_system_schematic(ax1)

        # Plot 2: Membrane deflection profile
        ax2 = self.figure.add_subplot(gs[1, 0])
        self.plot_membrane_profile(ax2)

        # Plot 3: Sensitivity vs pressure
        ax3 = self.figure.add_subplot(gs[1, 1])
        self.plot_sensitivity_curve(ax3)

        # Plot 4: SNR vs pressure
        ax4 = self.figure.add_subplot(gs[2, 0])
        self.plot_snr_curve(ax4)

        # Plot 5: Parameter sensitivity map
        ax5 = self.figure.add_subplot(gs[2, 1])
        self.plot_parameter_space(ax5)

        # Title
        self.figure.suptitle(
            f"Pressure: {self.params['pressure_pa']:.2e} Pa | "
            f"Signal: {self.results['displacement_pixels']:.3f} px | "
            f"SNR: {self.results['snr_db']:.1f} dB | "
            f"{self.results['regime']}",
            fontsize=11, fontweight='bold'
        )

        self.canvas.draw()

    def plot_system_schematic(self, ax):
        """Draw system schematic."""
        ax.clear()
        ax.set_aspect('equal')

        R = self.results['radius_m']
        delta = self.results['center_deflection_m']
        r_off = self.results['laser_offset_m']

        # Membrane with exaggerated deflection
        x = np.linspace(-R, R, 100)
        scale = min(0.1 / max(abs(delta), 1e-12), 1e6)
        y_membrane = -delta * (1 - (x ** 2 / R ** 2)) * scale

        ax.plot(x * 1000, y_membrane * 1000, 'b-', linewidth=2.5, label='Membrane')
        ax.fill_between(x * 1000, y_membrane * 1000, -20, color='lightblue', alpha=0.3)

        # Clamped edges
        ax.plot([-R * 1000, -R * 1000], [-20, 0], 'k-', linewidth=3)
        ax.plot([R * 1000, R * 1000], [-20, 0], 'k-', linewidth=3)

        # Laser beam
        laser_pos = r_off * 1000
        y_surf = -delta * (1 - (r_off ** 2 / R ** 2)) * scale * 1000
        ax.arrow(laser_pos, 50, 0, y_surf - 50, head_width=1, head_length=2,
                 fc='red', ec='red', linewidth=2)
        ax.text(laser_pos + 1, 55, 'Laser', fontsize=9, color='red')

        # Reflected beam
        slope = self.results['slope_angle_rad']
        dx_reflect = 20 * np.sin(2 * slope * scale)
        dy_reflect = 20 * np.cos(2 * slope * scale)
        ax.arrow(laser_pos, y_surf, dx_reflect, dy_reflect,
                 head_width=1, head_length=2, fc='orange', ec='orange',
                 linewidth=2, linestyle='--')

        # Detector
        det_y = 70
        ax.plot([-15, 15], [det_y, det_y], 'k-', linewidth=3)
        spot_x = self.results['spot_displacement_m'] * 1e3
        ax.plot(spot_x, det_y, 'ro', markersize=8)
        ax.text(spot_x + 2, det_y + 5, f'{spot_x:.2f} mm', fontsize=8)

        ax.set_xlim(-R * 1200, R * 1200)
        ax.set_ylim(-30, 80)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Height (mm, exaggerated)")
        ax.set_title("Autocollimator Schematic", fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_membrane_profile(self, ax):
        """Plot membrane deflection profile."""
        ax.clear()

        R = self.results['radius_m']
        delta = self.results['center_deflection_m']
        r_off = self.results['laser_offset_m']

        r = np.linspace(-R, R, 200)
        z = delta * (1 - (r ** 2 / R ** 2))

        ax.plot(r * 1000, z * 1e9, 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)

        # Mark laser position
        z_laser = delta * (1 - (r_off ** 2 / R ** 2))
        ax.plot(r_off * 1000, z_laser * 1e9, 'ro', markersize=8, label='Laser spot')

        # Draw slope tangent
        slope = self.results['slope_angle_rad']
        tan_len = R * 0.3
        ax.plot([r_off * 1000 - tan_len * 500, r_off * 1000 + tan_len * 500],
                [z_laser * 1e9 + slope * tan_len * 500 * 1e9,
                 z_laser * 1e9 - slope * tan_len * 500 * 1e9],
                'g--', linewidth=1.5, label=f'Slope: {slope * 1e6:.2f} μrad')

        ax.set_xlabel("Radial Position (mm)")
        ax.set_ylabel("Deflection (nm)")
        ax.set_title(f"Membrane Profile: δ₀ = {delta * 1e9:.2f} nm", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_sensitivity_curve(self, ax):
        """Plot pixel response vs pressure."""
        ax.clear()

        pressures = np.logspace(-6, 6, 150)
        pixels = []

        # Calculate response for range of pressures
        R = self.results['radius_m']
        T_eff = self.results['effective_tension']
        r_off = self.results['laser_offset_m']
        f = self.params['focal_length_m']
        pix_size = self.params['pixel_size_um'] * 1e-6

        for P in pressures:
            if T_eff > 0:
                delta = (P * R ** 2) / (4 * T_eff)
                slope = abs((2 * delta * r_off) / R ** 2) if R > 0 else 0
                spot = 2 * slope * f
                px = spot / pix_size if pix_size > 0 else 0
            else:
                px = 0
            pixels.append(px)

        ax.loglog(pressures, pixels, 'b-', linewidth=2)

        # Mark current point
        ax.plot(self.params['pressure_pa'], self.results['displacement_pixels'],
                'ro', markersize=8, label='Current')

        # Detection threshold
        ax.axhline(0.1, color='r', linestyle='--', alpha=0.5, label='0.1 px threshold')

        ax.set_xlabel("Pressure (Pa)")
        ax.set_ylabel("Pixel Displacement")
        ax.set_title("Sensor Response Curve", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    def plot_snr_curve(self, ax):
        """Plot SNR vs pressure."""
        ax.clear()

        pressures = np.logspace(-6, 6, 150)
        snr_vals = []

        R = self.results['radius_m']
        T_eff = self.results['effective_tension']
        r_off = self.results['laser_offset_m']
        f = self.params['focal_length_m']
        pix_size = self.params['pixel_size_um'] * 1e-6
        noise = self.params['camera_noise_pixels']

        for P in pressures:
            if T_eff > 0 and noise > 0:
                delta = (P * R ** 2) / (4 * T_eff)
                slope = abs((2 * delta * r_off) / R ** 2) if R > 0 else 0
                spot = 2 * slope * f
                px = spot / pix_size if pix_size > 0 else 0
                snr = px / noise
            else:
                snr = 0
            snr_vals.append(max(snr, 1e-6))

        ax.loglog(pressures, snr_vals, 'g-', linewidth=2)

        # Mark current
        ax.plot(self.params['pressure_pa'], max(self.results['snr_linear'], 1e-6),
                'ro', markersize=8, label='Current')

        # SNR thresholds
        ax.axhline(1, color='r', linestyle='--', alpha=0.5, label='SNR = 1')
        ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='SNR = 10')

        ax.set_xlabel("Pressure (Pa)")
        ax.set_ylabel("Signal-to-Noise Ratio")
        ax.set_title("SNR vs Pressure", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([1e-3, 1e5])

    def plot_parameter_space(self, ax):
        """Plot sensitivity map vs diameter and tension."""
        ax.clear()

        diameters = np.linspace(5, 100, 40)
        tensions = np.logspace(-1, 1, 40)

        D_grid, T_grid = np.meshgrid(diameters, tensions)
        sens_grid = np.zeros_like(D_grid)

        h = self.params['membrane_thickness_um'] * 1e-6
        E = self.params['youngs_modulus_GPa'] * 1e9
        nu = self.params['poissons_ratio']
        offset_pct = self.params['radial_offset_percent'] / 100
        f = self.params['focal_length_m']
        pix_size = self.params['pixel_size_um'] * 1e-6

        D_bend = (E * h ** 3) / (12 * (1 - nu ** 2))

        for i in range(D_grid.shape[0]):
            for j in range(D_grid.shape[1]):
                R = (D_grid[i, j] / 1000) / 2
                K_bend = 4 * D_bend / (R ** 2)
                T_eff = T_grid[i, j] + K_bend
                r_off = offset_pct * R

                if T_eff > 0 and pix_size > 0:
                    delta = (1.0 * R ** 2) / (4 * T_eff)  # For 1 Pa
                    slope = abs((2 * delta * r_off) / R ** 2) if R > 0 else 0
                    spot = 2 * slope * f
                    sens = spot / pix_size
                else:
                    sens = 0

                sens_grid[i, j] = sens

        im = ax.contourf(D_grid, T_grid, sens_grid, levels=30, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Sensitivity (px/Pa)')

        # Mark current config
        ax.plot(self.params['membrane_diameter_mm'],
                self.params['membrane_tension_Nm'],
                'rx', markersize=10, markeredgewidth=2)

        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("Tension (N/m)")
        ax.set_yscale('log')
        ax.set_title("Sensitivity Map", fontweight='bold')
        ax.grid(True, alpha=0.3)

    def update_results_text(self):
        """Update results display."""
        r = self.results
        p = self.params

        html = "<h3>Analysis Results</h3>"
        html += "<table style='width:100%; border-collapse: collapse;'>"

        # Pressure
        html += "<tr style='background: #e8f4f8;'><td colspan='3'><b>Applied Pressure</b></td></tr>"
        html += f"<tr><td>Value</td><td>{r['pressure_pa']:.3e}</td><td>Pa</td></tr>"
        html += f"<tr><td>Value</td><td>{r['pressure_pa'] * 1e6:.1f}</td><td>μPa</td></tr>"

        # Membrane response
        html += "<tr style='background: #e8f4f8;'><td colspan='3'><b>Membrane Response</b></td></tr>"
        html += f"<tr><td>Center Deflection</td><td>{r['center_deflection_nm']:.3f}</td><td>nm</td></tr>"
        html += f"<tr><td>Slope at Laser</td><td>{r['slope_angle_urad']:.3f}</td><td>μrad</td></tr>"
        html += f"<tr><td>Regime</td><td colspan='2'>{r['regime']}</td></tr>"
        html += f"<tr><td>Effective Tension</td><td>{r['effective_tension']:.3f}</td><td>N/m</td></tr>"

        # Optical
        html += "<tr style='background: #e8f4f8;'><td colspan='3'><b>Optical Signal</b></td></tr>"
        html += f"<tr><td>Reflected Angle</td><td>{r['reflected_angle_rad'] * 1e6:.3f}</td><td>μrad</td></tr>"
        html += f"<tr><td>Spot Displacement</td><td>{r['spot_displacement_um']:.3f}</td><td>μm</td></tr>"
        html += f"<tr><td><b>Pixel Displacement</b></td><td><b>{r['displacement_pixels']:.4f}</b></td><td><b>px</b></td></tr>"

        # Performance
        html += "<tr style='background: #e8f4f8;'><td colspan='3'><b>Performance</b></td></tr>"
        html += f"<tr><td>Sensitivity</td><td>{r['sensitivity_px_per_pa']:.3e}</td><td>px/Pa</td></tr>"
        html += f"<tr><td>Deflection Sensitivity</td><td>{r['sensitivity_nm_per_pa']:.3f}</td><td>nm/Pa</td></tr>"
        html += f"<tr><td>SNR (linear)</td><td>{r['snr_linear']:.2f}</td><td></td></tr>"
        html += f"<tr><td><b>SNR (dB)</b></td><td><b>{r['snr_db']:.1f}</b></td><td><b>dB</b></td></tr>"

        # Detection limits
        html += "<tr style='background: #e8f4f8;'><td colspan='3'><b>Detection Capability</b></td></tr>"
        html += f"<tr><td>Min Detectable (0.1px)</td><td>{r['min_detectable_pa']:.2e}</td><td>Pa</td></tr>"
        html += f"<tr><td>Min Detectable</td><td>{r['min_detectable_upa']:.1f}</td><td>μPa</td></tr>"

        html += "</table>"

        # Interpretation
        html += "<p style='margin-top:10px; padding:10px; background:#f0f0f0; border-left:4px solid #4CAF50;'>"

        if r['snr_db'] > 20:
            html += "✓ <b>Excellent detection:</b> SNR > 20 dB"
        elif r['snr_db'] > 10:
            html += "✓ <b>Good detection:</b> SNR > 10 dB"
        elif r['snr_db'] > 0:
            html += "⚠ <b>Marginal detection:</b> SNR between 0-10 dB"
        else:
            html += "✗ <b>Below noise floor:</b> SNR < 0 dB"

        html += "<br><br><b>Can detect:</b><br>"

        ref_levels = [
            ("Ambient ocean noise", 50e-6),
            ("Whale calls", 1e-3),
            ("Ship noise", 1.0),
            ("Sonar pulse", 10e3)
        ]

        for name, level in ref_levels:
            if r['min_detectable_pa'] <= level:
                html += f"✓ {name} ({level * 1e6:.0f} μPa)<br>"
            else:
                html += f"✗ {name} ({level * 1e6:.0f} μPa)<br>"

        html += "</p>"

        self.results_display.setHtml(html)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MylarHydrophoneSimulator()
    window.show()
    sys.exit(app.exec())