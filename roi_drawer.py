#!/usr/bin/env python3
"""
Unified Interactive ROI Drawing Tool
=====================================

Automatic mode detection based on --regions flag:

SINGLE-REGION MODE (no --regions):
  • All polygons combine into one region (binary mask)
  • Perfect for bilateral organs as one object
  
MULTI-REGION MODE (with --regions):
  • Draw multiple labeled regions independently  
  • Perfect for distinct anatomical structures

Both modes include:
  • Full undo/redo functionality
  • Zoom/pan controls
  • No matplotlib toolbar (cleaner interface)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.path import Path as MplPath
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys



class SingleRegionROIDrawer:
    """Interactive ROI drawing tool"""
    
    def __init__(self, image: np.ndarray, title: str = "Draw ROI"):
        """
        Initialize ROI drawer
        
        Parameters:
        -----------
        image : ndarray
            2D anatomical image to draw on
        title : str
            Window title
        """
        self.image = image
        self.title = title
        
        # ROI state
        self.roi_mask = np.zeros_like(image, dtype=bool)
        self.current_polygon = []
        self.polygon_closed = False
        
        # History for undo/redo
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        # Zoom/pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = None
        
        # Drawing state
        self.drawing = False
        self.mode = 'draw'  # 'draw' or 'pan'
        
        # Setup figure
        self.setup_figure()
        
        # Save initial state
        self.save_state()
        
    def setup_figure(self):
        """Setup matplotlib figure with controls"""
        # Disable toolbar to avoid confusion with PNG saving
        import matplotlib
        matplotlib.rcParams['toolbar'] = 'None'
        
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main image axes
        self.ax_image = plt.axes([0.05, 0.25, 0.75, 0.70])
        
        # Display image
        vmin, vmax = np.percentile(self.image[self.image > 0], [1, 99])
        self.im = self.ax_image.imshow(
            self.image, 
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        self.ax_image.set_title(self.title, fontsize=14, fontweight='bold')
        
        # Overlay for ROI
        self.roi_overlay = self.ax_image.imshow(
            np.ma.masked_where(~self.roi_mask, self.roi_mask),
            cmap='Reds',
            alpha=0.5,
            vmin=0,
            vmax=1
        )
        
        # Polygon line
        self.polygon_line, = self.ax_image.plot([], [], 'r-', linewidth=2)
        self.polygon_points, = self.ax_image.plot([], [], 'ro', markersize=6)
        
        # Status text
        self.status_text = self.ax_image.text(
            0.02, 0.98, 
            'Click to draw polygon. Close polygon to fill.',
            transform=self.ax_image.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10
        )
        
        # Buttons
        button_width = 0.10
        button_height = 0.04
        button_spacing = 0.01
        
        # Row 1: Drawing controls
        y_row1 = 0.15
        self.btn_undo = Button(
            plt.axes([0.05, y_row1, button_width, button_height]),
            'Undo (Z)'
        )
        self.btn_undo.on_clicked(self.undo)
        
        self.btn_redo = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row1, button_width, button_height]),
            'Redo (Y)'
        )
        self.btn_redo.on_clicked(self.redo)
        
        self.btn_clear = Button(
            plt.axes([0.05 + 2*(button_width + button_spacing), y_row1, button_width, button_height]),
            'Clear (C)'
        )
        self.btn_clear.on_clicked(self.clear)
        
        self.btn_close_poly = Button(
            plt.axes([0.05 + 3*(button_width + button_spacing), y_row1, button_width, button_height]),
            'Close Polygon'
        )
        self.btn_close_poly.on_clicked(self.close_polygon)
        
        # Row 2: Zoom/view controls
        y_row2 = 0.09
        self.btn_zoom_in = Button(
            plt.axes([0.05, y_row2, button_width, button_height]),
            'Zoom In (+)'
        )
        self.btn_zoom_in.on_clicked(lambda x: self.zoom(1.5))
        
        self.btn_zoom_out = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row2, button_width, button_height]),
            'Zoom Out (-)'
        )
        self.btn_zoom_out.on_clicked(lambda x: self.zoom(0.67))
        
        self.btn_reset_view = Button(
            plt.axes([0.05 + 2*(button_width + button_spacing), y_row2, button_width, button_height]),
            'Reset View (R)'
        )
        self.btn_reset_view.on_clicked(self.reset_view)
        
        self.btn_pan_mode = Button(
            plt.axes([0.05 + 3*(button_width + button_spacing), y_row2, button_width, button_height]),
            'Pan Mode (P)'
        )
        self.btn_pan_mode.on_clicked(self.toggle_pan_mode)
        
        # Row 3: Save/Exit
        y_row3 = 0.03
        self.btn_save = Button(
            plt.axes([0.05, y_row3, button_width, button_height]),
            'Save ROI (S)'
        )
        self.btn_save.on_clicked(self.save_roi)
        
        self.btn_done = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row3, button_width*1.5, button_height]),
            'Done & Exit (Enter)'
        )
        self.btn_done.on_clicked(self.done)
        
        # Info panel
        info_text = """
INSTRUCTIONS:
• Click to add points to polygon
• Close polygon to fill area
• Undo/Redo to fix mistakes
• Zoom/Pan to see details
• Save when done

KEYBOARD SHORTCUTS:
Z - Undo
Y - Redo  
C - Clear all
R - Reset view
P - Pan mode
+ - Zoom in
- - Zoom out
S - Save
Enter - Done
"""
        self.ax_info = plt.axes([0.82, 0.25, 0.17, 0.70])
        self.ax_info.text(0.05, 0.95, info_text, 
                         transform=self.ax_info.transAxes,
                         verticalalignment='top',
                         fontsize=9,
                         family='monospace')
        self.ax_info.axis('off')
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def save_state(self):
        """Save current state to history"""
        # Remove any redo history
        self.history = self.history[:self.history_index + 1]
        
        # Add current state
        state = {
            'roi_mask': self.roi_mask.copy(),
            'polygon': self.current_polygon.copy(),
            'polygon_closed': self.polygon_closed
        }
        self.history.append(state)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.history_index = len(self.history) - 1
        
    def restore_state(self, index: int):
        """Restore state from history"""
        if 0 <= index < len(self.history):
            state = self.history[index]
            self.roi_mask = state['roi_mask'].copy()
            self.current_polygon = state['polygon'].copy()
            self.polygon_closed = state['polygon_closed']
            self.history_index = index
            self.update_display()
            
    def undo(self, event=None):
        """Undo last action"""
        if self.history_index > 0:
            self.restore_state(self.history_index - 1)
            self.update_status("Undone")
        else:
            self.update_status("Nothing to undo")
            
    def redo(self, event=None):
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.restore_state(self.history_index + 1)
            self.update_status("Redone")
        else:
            self.update_status("Nothing to redo")
            
    def clear(self, event=None):
        """Clear all ROIs"""
        self.roi_mask = np.zeros_like(self.image, dtype=bool)
        self.current_polygon = []
        self.polygon_closed = False
        self.save_state()
        self.update_display()
        self.update_status("Cleared all ROIs")
        
    def close_polygon(self, event=None):
        """Close and fill current polygon"""
        if len(self.current_polygon) >= 3:
            # Create polygon path
            path = MplPath(self.current_polygon)
            
            # Create grid of points
            height, width = self.image.shape
            y, x = np.mgrid[:height, :width]
            points = np.vstack((x.ravel(), y.ravel())).T
            
            # Check which points are inside polygon
            mask = path.contains_points(points)
            mask = mask.reshape((height, width))
            
            # Add to ROI mask
            self.roi_mask |= mask
            
            # Clear polygon
            self.current_polygon = []
            self.polygon_closed = False
            
            # Save state
            self.save_state()
            self.update_display()
            self.update_status(f"Polygon filled. {np.sum(self.roi_mask)} pixels in ROI")
        else:
            self.update_status("Need at least 3 points to close polygon")
            
    def zoom(self, factor: float):
        """Zoom in/out"""
        self.zoom_level *= factor
        self.zoom_level = np.clip(self.zoom_level, 0.5, 10.0)
        self.apply_view()
        self.update_status(f"Zoom: {self.zoom_level:.1f}x")
        
    def reset_view(self, event=None):
        """Reset zoom and pan"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.apply_view()
        self.update_status("View reset")
        
    def toggle_pan_mode(self, event=None):
        """Toggle between draw and pan mode"""
        self.mode = 'pan' if self.mode == 'draw' else 'draw'
        mode_text = 'PAN' if self.mode == 'pan' else 'DRAW'
        self.update_status(f"Mode: {mode_text}")
        
    def apply_view(self):
        """Apply current zoom and pan to axes"""
        height, width = self.image.shape
        
        # Calculate view bounds
        center_x = width / 2 + self.pan_offset[0]
        center_y = height / 2 + self.pan_offset[1]
        
        view_width = width / self.zoom_level
        view_height = height / self.zoom_level
        
        x_min = center_x - view_width / 2
        x_max = center_x + view_width / 2
        y_min = center_y - view_height / 2
        y_max = center_y + view_height / 2
        
        self.ax_image.set_xlim(x_min, x_max)
        self.ax_image.set_ylim(y_max, y_min)  # Inverted y
        
        self.fig.canvas.draw_idle()
        
    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax_image:
            return
            
        if event.button == 1:  # Left click
            if self.mode == 'draw':
                # Add point to polygon
                x, y = int(round(event.xdata)), int(round(event.ydata))
                
                # Check bounds
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    self.current_polygon.append([x, y])
                    self.update_display()
                    self.update_status(f"Point {len(self.current_polygon)} added")
            elif self.mode == 'pan':
                # Start panning
                self.panning = True
                self.pan_start = [event.xdata, event.ydata]
                
    def on_release(self, event):
        """Handle mouse release"""
        if event.button == 1:
            self.panning = False
            self.pan_start = None
            
    def on_motion(self, event):
        """Handle mouse motion"""
        if self.panning and event.inaxes == self.ax_image and self.pan_start:
            # Calculate pan delta
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            
            # Update offset
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            
            # Apply view
            self.apply_view()
            
            # Update pan start
            self.pan_start = [event.xdata, event.ydata]
            
    def on_scroll(self, event):
        """Handle scroll wheel for zoom"""
        if event.inaxes == self.ax_image:
            if event.button == 'up':
                self.zoom(1.2)
            elif event.button == 'down':
                self.zoom(0.83)
                
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'z':
            self.undo()
        elif event.key == 'y':
            self.redo()
        elif event.key == 'c':
            self.clear()
        elif event.key == 'r':
            self.reset_view()
        elif event.key == 'p':
            self.toggle_pan_mode()
        elif event.key == '+' or event.key == '=':
            self.zoom(1.5)
        elif event.key == '-' or event.key == '_':
            self.zoom(0.67)
        elif event.key == 's':
            self.save_roi()
        elif event.key == 'enter':
            self.done()
            
    def update_display(self):
        """Update display with current ROI"""
        # Update ROI overlay
        self.roi_overlay.set_data(
            np.ma.masked_where(~self.roi_mask, self.roi_mask)
        )
        
        # Update polygon
        if self.current_polygon:
            poly = np.array(self.current_polygon)
            self.polygon_line.set_data(poly[:, 0], poly[:, 1])
            self.polygon_points.set_data(poly[:, 0], poly[:, 1])
        else:
            self.polygon_line.set_data([], [])
            self.polygon_points.set_data([], [])
            
        self.fig.canvas.draw_idle()
        
    def update_status(self, message: str):
        """Update status message"""
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()
        
    def save_roi(self, event=None):
        """Save ROI (to be implemented by caller)"""
        n_pixels = np.sum(self.roi_mask)
        self.update_status(f"ROI ready to save: {n_pixels} pixels")
        
    def done(self, event=None):
        """Close window"""
        plt.close(self.fig)
        
    def show(self):
        """Show the drawing interface"""
        plt.show()
        return self.roi_mask

class MultiRegionROIDrawer:
    """Interactive multi-region ROI drawing tool"""
    
    # Color palette for regions (distinct, colorblind-friendly)
    REGION_COLORS = [
        '#e41a1c',  # Red - Region 1
        '#377eb8',  # Blue - Region 2
        '#4daf4a',  # Green - Region 3
        '#984ea3',  # Purple - Region 4
        '#ff7f00',  # Orange - Region 5
        '#ffff33',  # Yellow - Region 6
        '#a65628',  # Brown - Region 7
        '#f781bf',  # Pink - Region 8
        '#999999',  # Gray - Region 9
    ]
    
    def __init__(self, image: np.ndarray, region_names: Optional[List[str]] = None,
                 title: str = "Draw Multi-Region ROI"):
        """
        Initialize multi-region ROI drawer
        
        Parameters:
        -----------
        image : ndarray
            2D anatomical image to draw on
        region_names : list of str, optional
            Names for each region (e.g., ['cortex', 'medulla', 'papilla'])
            If None, regions named Region 1, Region 2, etc.
        title : str
            Window title
        """
        self.image = image
        self.title = title
        
        # Region names
        if region_names is None:
            self.region_names = [f"Region {i+1}" for i in range(9)]
        else:
            self.region_names = region_names[:9]  # Max 9 regions
            # Pad if needed
            while len(self.region_names) < 9:
                self.region_names.append(f"Region {len(self.region_names)+1}")
        
        # Multi-region state
        self.regions = {}  # {region_id: {'mask': ndarray, 'name': str, 'polygons': list}}
        self.current_region_id = 1
        self.current_polygon = []
        self.max_regions = len(self.REGION_COLORS)
        
        # History for undo/redo (per region)
        self.region_history = {}  # {region_id: [states]}
        self.region_history_index = {}  # {region_id: current_index}
        self.max_history = 50
        
        # Zoom/pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = None
        
        # Drawing state
        self.mode = 'draw'  # 'draw' or 'pan'
        
        # Setup figure
        self.setup_figure()
        
    def setup_figure(self):
        """Setup matplotlib figure with multi-region controls"""
        # Disable toolbar to avoid confusion with PNG saving
        import matplotlib
        matplotlib.rcParams['toolbar'] = 'None'
        
        self.fig = plt.figure(figsize=(18, 11))
        
        # Main image axes
        self.ax_image = plt.axes([0.05, 0.30, 0.65, 0.65])
        
        # Display image
        vmin, vmax = np.percentile(self.image[self.image > 0], [1, 99])
        self.im = self.ax_image.imshow(
            self.image, 
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        self.ax_image.set_title(self.title, fontsize=14, fontweight='bold')
        
        # Overlay for all regions (will be updated dynamically)
        self.region_overlay = self.ax_image.imshow(
            np.zeros_like(self.image),
            alpha=0.4,
            vmin=0,
            vmax=self.max_regions,
            cmap='tab10'
        )
        
        # Current polygon line and points
        color = self.REGION_COLORS[self.current_region_id - 1]
        self.polygon_line, = self.ax_image.plot([], [], color=color, linewidth=2.5)
        self.polygon_points, = self.ax_image.plot([], [], 'o', color=color, markersize=7)
        
        # Status text
        self.status_text = self.ax_image.text(
            0.02, 0.98, 
            f'Drawing {self.region_names[0]}. Click to add points.',
            transform=self.ax_image.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10,
            fontweight='bold'
        )
        
        # Region selector (radio buttons)
        self.ax_region_selector = plt.axes([0.72, 0.55, 0.12, 0.40])
        self.ax_region_selector.set_title('Select Region:', fontweight='bold')
        region_labels = [f"{i+1}. {name}" for i, name in enumerate(self.region_names[:6])]
        self.radio_regions = RadioButtons(self.ax_region_selector, region_labels)
        self.radio_regions.on_clicked(self.switch_region)
        
        # Buttons - Row 1: Current region controls
        button_width = 0.10
        button_height = 0.04
        button_spacing = 0.01
        y_row1 = 0.20
        
        self.btn_close_poly = Button(
            plt.axes([0.05, y_row1, button_width, button_height]),
            'Close Polygon\n(Space)'
        )
        self.btn_close_poly.on_clicked(self.close_polygon)
        
        self.btn_undo = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row1, button_width, button_height]),
            'Undo\n(U)'
        )
        self.btn_undo.on_clicked(self.undo)
        
        self.btn_redo = Button(
            plt.axes([0.05 + 2*(button_width + button_spacing), y_row1, button_width, button_height]),
            'Redo\n(R)'
        )
        self.btn_redo.on_clicked(self.redo)
        
        self.btn_delete_region = Button(
            plt.axes([0.05 + 3*(button_width + button_spacing), y_row1, button_width, button_height]),
            'Delete Region\n(D)'
        )
        self.btn_delete_region.on_clicked(self.delete_current_region)
        
        # Row 2: View controls
        y_row2 = 0.13
        
        self.btn_zoom_in = Button(
            plt.axes([0.05, y_row2, button_width, button_height]),
            'Zoom In\n(+)'
        )
        self.btn_zoom_in.on_clicked(lambda x: self.zoom(1.5))
        
        self.btn_zoom_out = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row2, button_width, button_height]),
            'Zoom Out\n(-)'
        )
        self.btn_zoom_out.on_clicked(lambda x: self.zoom(0.67))
        
        self.btn_reset_view = Button(
            plt.axes([0.05 + 2*(button_width + button_spacing), y_row2, button_width, button_height]),
            'Reset View\n(V)'
        )
        self.btn_reset_view.on_clicked(self.reset_view)
        
        self.btn_pan_mode = Button(
            plt.axes([0.05 + 3*(button_width + button_spacing), y_row2, button_width, button_height]),
            'Pan Mode\n(P)'
        )
        self.btn_pan_mode.on_clicked(self.toggle_pan_mode)
        
        # Row 3: Save/Exit
        y_row3 = 0.06
        
        self.btn_clear_all = Button(
            plt.axes([0.05, y_row3, button_width, button_height]),
            'Clear All\n(C)'
        )
        self.btn_clear_all.on_clicked(self.clear_all)
        
        self.btn_save = Button(
            plt.axes([0.05 + button_width + button_spacing, y_row3, button_width*1.2, button_height]),
            'Save & Exit\n(S)'
        )
        self.btn_save.on_clicked(self.save_and_exit)
        
        self.btn_quit = Button(
            plt.axes([0.05 + 2.2*(button_width + button_spacing), y_row3, button_width*1.2, button_height]),
            'Quit\n(Q)'
        )
        self.btn_quit.on_clicked(self.quit_without_save)
        
        # Info panel
        info_text = f"""
{'='*30}
MULTI-REGION ROI DRAWER
{'='*30}

DRAWING:
• Click: Add point to polygon
• Space: Close & fill polygon
• U: Undo last point/polygon
• R: Redo
• D: Delete current region

REGION SELECTION:
• 1-9: Switch to region 1-9
• Radio buttons: Click to select

VIEW:
• +/-: Zoom in/out
• P: Pan mode (drag image)
• V: Reset view
• Scroll: Zoom

SAVE:
• S: Save all regions & exit
• Q: Quit without saving

CURRENT REGIONS:
"""
        self.ax_info = plt.axes([0.72, 0.06, 0.27, 0.45])
        self.info_text = self.ax_info.text(
            0.02, 0.98, 
            info_text,
            transform=self.ax_info.transAxes,
            verticalalignment='top',
            fontsize=9,
            family='monospace'
        )
        self.ax_info.axis('off')
        
        # Region summary (will update dynamically)
        self.update_region_summary()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def switch_region(self, label):
        """Switch to different region"""
        # Extract region number from label (format: "1. Region Name")
        try:
            region_num = int(label.split('.')[0])
            if 1 <= region_num <= self.max_regions:
                # Save current polygon if in progress
                if self.current_polygon:
                    response = input(f"\nCurrent polygon for {self.region_names[self.current_region_id-1]} not closed. "
                                   "Close it first? (y/n): ")
                    if response.lower() == 'y':
                        self.close_polygon()
                    else:
                        self.current_polygon = []
                
                self.current_region_id = region_num
                self.update_polygon_colors()
                self.update_display()
                self.update_status(f"Switched to {self.region_names[region_num-1]}")
        except (ValueError, IndexError):
            pass
            
    def save_point_state(self):
        """Save current polygon state (for undo of points)"""
        region_id = self.current_region_id
        
        # Initialize history for this region if needed
        if region_id not in self.region_history:
            self.region_history[region_id] = []
            self.region_history_index[region_id] = -1
        
        # Remove any redo history
        idx = self.region_history_index[region_id]
        self.region_history[region_id] = self.region_history[region_id][:idx + 1]
        
        # Save current state including in-progress polygon
        if region_id in self.regions:
            state = {
                'mask': self.regions[region_id]['mask'].copy(),
                'polygons': [p.copy() for p in self.regions[region_id].get('polygons', [])],
                'current_polygon': self.current_polygon.copy()
            }
        else:
            state = {
                'mask': np.zeros_like(self.image, dtype=bool),
                'polygons': [],
                'current_polygon': self.current_polygon.copy()
            }
        
        self.region_history[region_id].append(state)
        
        # Limit history size
        if len(self.region_history[region_id]) > self.max_history:
            self.region_history[region_id] = self.region_history[region_id][-self.max_history:]
        
        self.region_history_index[region_id] = len(self.region_history[region_id]) - 1
        
    def save_region_state(self):
        """Save current state of active region to history"""
        region_id = self.current_region_id
        
        # Initialize history for this region if needed
        if region_id not in self.region_history:
            self.region_history[region_id] = []
            self.region_history_index[region_id] = -1
        
        # Remove any redo history
        idx = self.region_history_index[region_id]
        self.region_history[region_id] = self.region_history[region_id][:idx + 1]
        
        # Save current state
        if region_id in self.regions:
            state = {
                'mask': self.regions[region_id]['mask'].copy(),
                'polygons': [p.copy() for p in self.regions[region_id].get('polygons', [])],
                'current_polygon': []  # Polygon closed, so clear it
            }
        else:
            state = {
                'mask': np.zeros_like(self.image, dtype=bool),
                'polygons': [],
                'current_polygon': []
            }
        
        self.region_history[region_id].append(state)
        
        # Limit history size
        if len(self.region_history[region_id]) > self.max_history:
            self.region_history[region_id] = self.region_history[region_id][-self.max_history:]
        
        self.region_history_index[region_id] = len(self.region_history[region_id]) - 1
        
    def undo(self, event=None):
        """Undo last action for current region"""
        region_id = self.current_region_id
        
        if region_id in self.region_history and self.region_history_index[region_id] > 0:
            self.region_history_index[region_id] -= 1
            idx = self.region_history_index[region_id]
            state = self.region_history[region_id][idx]
            
            # Restore state
            if region_id in self.regions:
                self.regions[region_id]['mask'] = state['mask'].copy()
                self.regions[region_id]['polygons'] = [p.copy() for p in state['polygons']]
            
            # Restore current polygon if present
            self.current_polygon = state.get('current_polygon', []).copy()
            
            self.update_display()
            self.update_status(f"Undone - {self.region_names[region_id-1]}")
        else:
            self.update_status("Nothing to undo")
            
    def redo(self, event=None):
        """Redo last undone action for current region"""
        region_id = self.current_region_id
        
        if (region_id in self.region_history and 
            self.region_history_index[region_id] < len(self.region_history[region_id]) - 1):
            self.region_history_index[region_id] += 1
            idx = self.region_history_index[region_id]
            state = self.region_history[region_id][idx]
            
            # Restore state
            if region_id in self.regions:
                self.regions[region_id]['mask'] = state['mask'].copy()
                self.regions[region_id]['polygons'] = [p.copy() for p in state['polygons']]
            
            # Restore current polygon if present
            self.current_polygon = state.get('current_polygon', []).copy()
            
            self.update_display()
            self.update_status(f"Redone - {self.region_names[region_id-1]}")
        else:
            self.update_status("Nothing to redo")
            
    def close_polygon(self, event=None):
        """Close and fill current polygon"""
        if len(self.current_polygon) >= 3:
            region_id = self.current_region_id
            
            # Initialize region if needed
            if region_id not in self.regions:
                self.regions[region_id] = {
                    'mask': np.zeros_like(self.image, dtype=bool),
                    'name': self.region_names[region_id - 1],
                    'polygons': []
                }
            
            # Create polygon path
            path = MplPath(self.current_polygon)
            
            # Create grid of points
            height, width = self.image.shape
            y, x = np.mgrid[:height, :width]
            points = np.vstack((x.ravel(), y.ravel())).T
            
            # Check which points are inside polygon
            mask = path.contains_points(points)
            mask = mask.reshape((height, width))
            
            # Add to region mask
            self.regions[region_id]['mask'] |= mask
            self.regions[region_id]['polygons'].append(self.current_polygon.copy())
            
            # Clear current polygon
            n_pixels = np.sum(self.regions[region_id]['mask'])
            self.current_polygon = []
            
            # Save state
            self.save_region_state()
            self.update_display()
            self.update_region_summary()
            self.update_status(f"Polygon added to {self.region_names[region_id-1]} ({n_pixels:,} pixels)")
        else:
            self.update_status("Need at least 3 points to close polygon")
            
    def delete_current_region(self, event=None):
        """Delete current region entirely"""
        region_id = self.current_region_id
        
        if region_id in self.regions:
            del self.regions[region_id]
            self.current_polygon = []
            
            # Clear history for this region
            if region_id in self.region_history:
                del self.region_history[region_id]
                del self.region_history_index[region_id]
            
            self.update_display()
            self.update_region_summary()
            self.update_status(f"Deleted {self.region_names[region_id-1]}")
        else:
            self.update_status(f"No {self.region_names[region_id-1]} to delete")
            
    def clear_all(self, event=None):
        """Clear all regions"""
        self.regions = {}
        self.current_polygon = []
        self.region_history = {}
        self.region_history_index = {}
        self.update_display()
        self.update_region_summary()
        self.update_status("Cleared all regions")
        
    def update_polygon_colors(self):
        """Update polygon line/point colors to match current region"""
        color = self.REGION_COLORS[self.current_region_id - 1]
        self.polygon_line.set_color(color)
        self.polygon_points.set_color(color)
        
    def zoom(self, factor: float):
        """Zoom in/out"""
        self.zoom_level *= factor
        self.zoom_level = np.clip(self.zoom_level, 0.5, 10.0)
        self.apply_view()
        self.update_status(f"Zoom: {self.zoom_level:.1f}x")
        
    def reset_view(self, event=None):
        """Reset zoom and pan"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.apply_view()
        self.update_status("View reset")
        
    def toggle_pan_mode(self, event=None):
        """Toggle between draw and pan mode"""
        self.mode = 'pan' if self.mode == 'draw' else 'draw'
        mode_text = 'PAN' if self.mode == 'pan' else 'DRAW'
        self.update_status(f"Mode: {mode_text}")
        
    def apply_view(self):
        """Apply current zoom and pan to axes"""
        height, width = self.image.shape
        
        # Calculate view bounds
        center_x = width / 2 + self.pan_offset[0]
        center_y = height / 2 + self.pan_offset[1]
        
        view_width = width / self.zoom_level
        view_height = height / self.zoom_level
        
        x_min = center_x - view_width / 2
        x_max = center_x + view_width / 2
        y_min = center_y - view_height / 2
        y_max = center_y + view_height / 2
        
        self.ax_image.set_xlim(x_min, x_max)
        self.ax_image.set_ylim(y_max, y_min)  # Inverted y
        
        self.fig.canvas.draw_idle()
        
    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax_image:
            return
            
        if event.button == 1:  # Left click
            if self.mode == 'draw':
                # Add point to polygon
                x, y = int(round(event.xdata)), int(round(event.ydata))
                
                # Check bounds
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    self.current_polygon.append([x, y])
                    
                    # Save state after adding point (for undo)
                    self.save_point_state()
                    
                    self.update_display()
                    region_name = self.region_names[self.current_region_id - 1]
                    self.update_status(f"{region_name}: Point {len(self.current_polygon)} added")
            elif self.mode == 'pan':
                # Start panning
                self.panning = True
                self.pan_start = [event.xdata, event.ydata]
                
    def on_release(self, event):
        """Handle mouse release"""
        if event.button == 1:
            self.panning = False
            self.pan_start = None
            
    def on_motion(self, event):
        """Handle mouse motion"""
        if self.panning and event.inaxes == self.ax_image and self.pan_start:
            # Calculate pan delta
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            
            # Update offset
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            
            # Apply view
            self.apply_view()
            
            # Update pan start
            self.pan_start = [event.xdata, event.ydata]
            
    def on_scroll(self, event):
        """Handle scroll wheel for zoom"""
        if event.inaxes == self.ax_image:
            if event.button == 'up':
                self.zoom(1.2)
            elif event.button == 'down':
                self.zoom(0.83)
                
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'u':
            self.undo()
        elif event.key == 'r':
            self.redo()
        elif event.key == 'c':
            self.clear_all()
        elif event.key == 'd':
            self.delete_current_region()
        elif event.key == 'v':
            self.reset_view()
        elif event.key == 'p':
            self.toggle_pan_mode()
        elif event.key == '+' or event.key == '=':
            self.zoom(1.5)
        elif event.key == '-' or event.key == '_':
            self.zoom(0.67)
        elif event.key == ' ':  # Space
            self.close_polygon()
        elif event.key == 's':
            self.save_and_exit()
        elif event.key == 'q':
            self.quit_without_save()
        elif event.key in '123456789':
            # Switch to region by number
            region_num = int(event.key)
            if region_num <= len(self.region_names):
                label = f"{region_num}. {self.region_names[region_num-1]}"
                self.radio_regions.set_active(region_num - 1)
                self.switch_region(label)
                
    def update_display(self):
        """Update display with all regions and current polygon"""
        # Create composite overlay with all regions
        composite = np.zeros_like(self.image, dtype=np.int32)
        
        for region_id, region_data in self.regions.items():
            composite[region_data['mask']] = region_id
        
        # Update region overlay
        self.region_overlay.set_data(
            np.ma.masked_where(composite == 0, composite)
        )
        
        # Update current polygon
        if self.current_polygon:
            poly = np.array(self.current_polygon)
            self.polygon_line.set_data(poly[:, 0], poly[:, 1])
            self.polygon_points.set_data(poly[:, 0], poly[:, 1])
        else:
            self.polygon_line.set_data([], [])
            self.polygon_points.set_data([], [])
            
        self.fig.canvas.draw_idle()
        
    def update_region_summary(self):
        """Update region summary in info panel"""
        info_base = f"""
{'='*30}
MULTI-REGION ROI DRAWER
{'='*30}

DRAWING:
• Click: Add point to polygon
• Space: Close & fill polygon
• U: Undo last point/polygon
• R: Redo
• D: Delete current region

REGION SELECTION:
• 1-9: Switch to region 1-9
• Radio buttons: Click to select

VIEW:
• +/-: Zoom in/out
• P: Pan mode (drag image)
• V: Reset view
• Scroll: Zoom

SAVE:
• S: Save all regions & exit
• Q: Quit without saving

CURRENT REGIONS:
"""
        
        if self.regions:
            region_info = []
            for region_id in sorted(self.regions.keys()):
                region_data = self.regions[region_id]
                n_pixels = np.sum(region_data['mask'])
                name = region_data['name']
                region_info.append(f"  {region_id}. {name}: {n_pixels:,} px")
            info_text = info_base + "\n".join(region_info)
        else:
            info_text = info_base + "  (none drawn yet)"
            
        self.info_text.set_text(info_text)
        self.fig.canvas.draw_idle()
        
    def update_status(self, message: str):
        """Update status message"""
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()
        
    def get_multi_label_mask(self) -> np.ndarray:
        """
        Get final multi-label mask
        
        Returns:
        --------
        mask : ndarray
            Integer mask where each pixel value indicates region ID (0 = background)
        """
        mask = np.zeros_like(self.image, dtype=np.int32)
        
        for region_id, region_data in self.regions.items():
            mask[region_data['mask']] = region_id
        
        return mask
        
    def save_and_exit(self, event=None):
        """Save and close"""
        self.should_save = True
        plt.close(self.fig)
        
    def quit_without_save(self, event=None):
        """Quit without saving"""
        self.should_save = False
        plt.close(self.fig)
        
    def show(self) -> Optional[np.ndarray]:
        """Show the drawing interface and return multi-label mask"""
        self.should_save = False
        plt.show()
        
        if self.should_save and self.regions:
            return self.get_multi_label_mask()
        else:
            return None


def pick_draw_image(prepared_dir: Path) -> Optional[Path]:
    """
    Show all _draw_*.npy variants in prepared_dir as a numbered grid,
    ask the user to pick one, and return its path.
    """
    draw_files = sorted(prepared_dir.glob("*_draw_*.npy"))
    if not draw_files:
        print(f"No _draw_*.npy files found in {prepared_dir}")
        print("Run  make_draw_refs.py <sample_id>  first.")
        return None

    # Load thumbnails
    imgs = []
    for f in draw_files:
        try:
            imgs.append((f, np.load(f).astype(np.float32)))
        except Exception:
            pass

    n = len(imgs)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (f, arr) in enumerate(imgs):
        ax = axes[i]
        vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
        ax.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        label = f.stem.split('_draw_', 1)[-1]
        ax.set_title(f"[{i+1}]  {label}", fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Select reference image for ROI drawing — {prepared_dir.name}\n"
        "Close this window, then type the number of your choice.",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.show(block=True)   # blocks until user closes the window

    # Terminal prompt
    print("\nAvailable images:")
    for i, (f, _) in enumerate(imgs):
        label = f.stem.split('_draw_', 1)[-1]
        print(f"  [{i+1}] {label}")
    while True:
        try:
            choice = int(input(f"\nEnter number (1–{n}): ").strip())
            if 1 <= choice <= n:
                chosen = imgs[choice - 1][0]
                print(f"Selected: {chosen.name}")
                return chosen
            print(f"  Please enter a number between 1 and {n}.")
        except (ValueError, EOFError):
            print("  Invalid input.")


def main():
    parser = argparse.ArgumentParser(
        description='Unified interactive ROI drawing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Direct — supply image path
  python roi_drawer.py kidney.npy -o mask.npy --regions left_kidney right_kidney

  # Pick mode — choose from all available draw variants
  python roi_drawer.py --pick processed/prepared/174229/ -o mask.npy --regions left_kidney right_kidney

Mode (single vs multi-region) is automatically detected from --regions.
        """
    )
    parser.add_argument('image', nargs='?', default=None,
                        help='Path to anatomical image (.npy) — omit when using --pick')
    parser.add_argument('--pick', metavar='PREPARED_DIR',
                        help='Directory of prepared sample: show all _draw_*.npy variants '
                             'and interactively select one before drawing')
    parser.add_argument('--output', '-o', required=True, help='Output path for ROI mask')
    parser.add_argument('--regions', '-r', nargs='+',
                        help='Region names for multi-region mode')
    parser.add_argument('--title', '-t', default='Draw ROI', help='Window title')

    args = parser.parse_args()

    # Resolve image path — either direct or via --pick
    if args.pick:
        chosen = pick_draw_image(Path(args.pick))
        if chosen is None:
            sys.exit(1)
        image_path = chosen
    elif args.image:
        image_path = Path(args.image)
    else:
        parser.error("Provide an image path or use --pick PREPARED_DIR")
    
    # Load image
    print(f"Loading image: {image_path}")
    image = np.load(image_path)
    print(f"Image shape: {image.shape}")
    
    # Detect mode and run appropriate drawer
    if args.regions:
        # MULTI-REGION MODE
        print(f"\n{'='*70}")
        print("MODE: MULTI-REGION")
        print(f"{'='*70}")
        print(f"Regions: {', '.join(args.regions)}")
        
        drawer = MultiRegionROIDrawer(image, args.regions, args.title)
        mask = drawer.show()
        
        if mask is not None:
            unique_regions = np.unique(mask[mask > 0])
            if len(unique_regions) > 0:
                np.save(args.output, mask)
                print(f"\n✓ Saved multi-region mask: {args.output}")
                print(f"\nRegions:")
                for region_id in unique_regions:
                    n_pixels = np.sum(mask == region_id)
                    name = args.regions[region_id-1] if region_id <= len(args.regions) else f"Region {region_id}"
                    print(f"  {region_id}. {name}: {n_pixels:,} pixels")
            else:
                print("\n⚠️  No regions drawn")
        else:
            print("\n⚠️  Quit without saving")
    else:
        # SINGLE-REGION MODE
        print(f"\n{'='*70}")
        print("MODE: SINGLE-REGION")
        print(f"{'='*70}")
        print("All polygons will combine into one region")
        
        drawer = SingleRegionROIDrawer(image, args.title)
        mask = drawer.show()
        
        if mask is not None and np.sum(mask) > 0:
            np.save(args.output, mask)
            print(f"\n✓ Saved mask: {args.output}")
            print(f"  Pixels: {np.sum(mask):,}")
        elif mask is not None:
            print("\n⚠️  No ROI drawn")
        else:
            print("\n⚠️  Quit without saving")


if __name__ == '__main__':
    main()
