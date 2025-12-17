#!/usr/bin/env python3
"""
Performance options for the Golf GUI
Includes settings for optimizing simulation performance
"""

import tkinter as tk
from tkinter import ttk


class PerformanceOptionsDialog:
    """Dialog for configuring simulation performance options"""

    def __init__(self, parent):
        self.parent = parent
        self.result = None

        # Default settings
        self.settings = {
            "disable_simscape_results": True,  # Default to enabled for performance
            "optimize_memory": True,
            "fast_restart": False,
        }

        self.create_dialog()

    def create_dialog(self):
        """Create the performance options dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Simulation Performance Options")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")

        # Create main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Performance Optimization Settings",
            font=("Arial", 12, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Simscape Results Explorer option
        self.simscape_var = tk.BooleanVar(
            value=self.settings["disable_simscape_results"]
        )
        simscape_check = ttk.Checkbutton(
            main_frame,
            text="Disable Simscape Results Explorer",
            variable=self.simscape_var,
            command=self.update_simscape_info,
        )
        simscape_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Info label for Simscape option
        self.simscape_info = ttk.Label(
            main_frame,
            text="✅ Provides ~5% speed improvement\n   Reduces memory usage during simulation",
            foreground="green",
        )
        self.simscape_info.grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 15)
        )

        # Memory optimization option
        self.memory_var = tk.BooleanVar(value=self.settings["optimize_memory"])
        memory_check = ttk.Checkbutton(
            main_frame, text="Optimize Memory Usage", variable=self.memory_var
        )
        memory_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

        memory_info = ttk.Label(
            main_frame,
            text="Reduces memory allocation during simulation",
            foreground="blue",
        )
        memory_info.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))

        # Fast restart option
        self.fast_restart_var = tk.BooleanVar(value=self.settings["fast_restart"])
        fast_restart_check = ttk.Checkbutton(
            main_frame, text="Enable Fast Restart", variable=self.fast_restart_var
        )
        fast_restart_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        fast_restart_info = ttk.Label(
            main_frame,
            text="Faster subsequent simulations (may use more memory)",
            foreground="blue",
        )
        fast_restart_info.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(20, 0))

        ok_button = ttk.Button(button_frame, text="OK", command=self.ok_clicked)
        ok_button.grid(row=0, column=0, padx=(0, 10))

        cancel_button = ttk.Button(
            button_frame, text="Cancel", command=self.cancel_clicked
        )
        cancel_button.grid(row=0, column=1)

        # Initialize info display
        self.update_simscape_info()

    def update_simscape_info(self):
        """Update the Simscape option info display"""
        if self.simscape_var.get():
            self.simscape_info.config(
                text="✅ Provides ~5% speed improvement\n   Reduces memory usage during simulation",
                foreground="green",
            )
        else:
            self.simscape_info.config(
                text="⚠️  Simscape Results Explorer enabled\n   May slow down simulation",
                foreground="orange",
            )

    def ok_clicked(self):
        """Handle OK button click"""
        self.settings = {
            "disable_simscape_results": self.simscape_var.get(),
            "optimize_memory": self.memory_var.get(),
            "fast_restart": self.fast_restart_var.get(),
        }
        self.result = self.settings
        self.dialog.destroy()

    def cancel_clicked(self):
        """Handle Cancel button click"""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Show the dialog and return the result"""
        self.dialog.wait_window()
        return self.result


def get_performance_options(parent):
    """Show performance options dialog and return settings"""
    dialog = PerformanceOptionsDialog(parent)
    return dialog.show()


def generate_matlab_performance_script(settings):
    """Generate MATLAB script with performance settings"""
    script_lines = []

    script_lines.append("% Performance optimization settings")
    script_lines.append("% Generated by Golf GUI")
    script_lines.append("")

    if settings["disable_simscape_results"]:
        script_lines.append(
            "% Disable Simscape Results Explorer for better performance"
        )
        script_lines.append("set_param(gcs, 'SimscapeLogType', 'none');")
        script_lines.append("")

    if settings["optimize_memory"]:
        script_lines.append("% Optimize memory usage")
        script_lines.append("set_param(gcs, 'MemoryReduction', 'on');")
        script_lines.append("")

    if settings["fast_restart"]:
        script_lines.append("% Enable fast restart")
        script_lines.append("set_param(gcs, 'FastRestart', 'on');")
        script_lines.append("")

    script_lines.append("% Apply settings")
    script_lines.append("apply_param_changes = true;")

    return "\n".join(script_lines)


if __name__ == "__main__":
    # Test the dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    settings = get_performance_options(root)

    if settings:
        print("Selected settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

        print("\nGenerated MATLAB script:")
        print(generate_matlab_performance_script(settings))
    else:
        print("Dialog cancelled")

    root.destroy()
