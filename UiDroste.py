#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import os
import json
import threading
from DrosteTransform import DrosteTransform


class DrosteGUI:
    CONFIG_FILE = "droste-config.json"

    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Droste Effect")
        self.root.geometry("2000x1400")

        self.transformer = None
        self.photo = None
        self.source_photo = None
        self.last_full_result = None
        self.source_image_path = None

        self.full_render_thread = None
        self.cancel_full_render = False

        self.params = {
            'inner_radius': tk.DoubleVar(value=25),
            'outer_radius': tk.DoubleVar(value=100),
            'periodicity': tk.DoubleVar(value=1.0),
            'strands': tk.IntVar(value=1),
            'zoom': tk.DoubleVar(value=1.0),
            'rotate': tk.DoubleVar(value=0.0),
            'x_shift': tk.DoubleVar(value=0.0),
            'y_shift': tk.DoubleVar(value=0.0),
            'x_center_shift': tk.DoubleVar(value=0.0),
            'y_center_shift': tk.DoubleVar(value=0.0),
            'starting_level': tk.IntVar(value=1),
            'num_levels': tk.IntVar(value=10),
            'level_frequency': tk.IntVar(value=1),
            'pole_rotation': tk.DoubleVar(value=90),
            'pole_long': tk.DoubleVar(value=0.0),
            'pole_lat': tk.DoubleVar(value=0.0),
            'fractal_points': tk.IntVar(value=1)
        }

        self.show_both_poles = tk.BooleanVar(value=False)
        self.tile_poles = tk.BooleanVar(value=False)
        self.hyper_droste = tk.BooleanVar(value=False)
        self.auto_periodicity = tk.BooleanVar(value=False)
        self.mirror_effect = tk.BooleanVar(value=False)
        self.untwist = tk.BooleanVar(value=False)
        self.no_transparency = tk.BooleanVar(value=False)
        self.external_transparency = tk.BooleanVar(value=False)
        self.do_not_flatten_transparency = tk.BooleanVar(value=False)
        self.show_grid = tk.BooleanVar(value=False)

        self.update_pending = None
        self.is_updating = False
        self.preview_max_size = 128

        self.setup_ui()
        self.load_last_image()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame, width=600)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=2)

        bool_frame = ttk.LabelFrame(control_frame, text="Options", padding=5)
        bool_frame.pack(pady=10, fill=tk.X)

        ttk.Checkbutton(bool_frame, text="Show Both Poles", variable=self.show_both_poles,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Tile Poles", variable=self.tile_poles, command=self.schedule_update).pack(
            anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Hyper Droste", variable=self.hyper_droste, command=self.schedule_update).pack(
            anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Auto Periodicity", variable=self.auto_periodicity,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Mirror Effect", variable=self.mirror_effect,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Untwist", variable=self.untwist, command=self.schedule_update).pack(
            anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="No Transparency", variable=self.no_transparency,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="External Transparency", variable=self.external_transparency,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Keep Transparency", variable=self.do_not_flatten_transparency,
                        command=self.schedule_update).pack(anchor=tk.W)
        ttk.Checkbutton(bool_frame, text="Show Grid", variable=self.show_grid,
                        command=self.schedule_update).pack(anchor=tk.W)

        params_config = [
            ('inner_radius', 0.01, 100.0, 0.01),
            ('outer_radius', 0.01, 100.0, 0.01),
            ('periodicity', -6.0, 6.0, 0.01),
            ('strands', -6, 6, 1),
            ('zoom', 0.1, 100.0, 0.1),
            ('rotate', -360, 360, 1),
            ('x_shift', -100.0, 100.0, 0.01),
            ('y_shift', -100.0, 100.0, 0.01),
            ('x_center_shift', -100.0, 100.0, 0.01),
            ('y_center_shift', -100.0, 100.0, 0.01),
            ('starting_level', 1, 20, 1),
            ('num_levels', 1, 20, 1),
            ('level_frequency', 1, 10, 1),
            ('pole_rotation', -180, 180, 1),
            ('pole_long', -100.0, 100.0, 0.01),
            ('pole_lat', -100.0, 100.0, 0.01),
            ('fractal_points', 1, 10, 1)
        ]

        label_map = {
            'inner_radius': 'inner rad',
            'outer_radius': 'outer rad',
            'periodicity': 'period',
            'strands': 'strands',
            'zoom': 'zoom',
            'rotate': 'rotate',
            'x_shift': 'x shift',
            'y_shift': 'y shift',
            'x_center_shift': 'xc shift',
            'y_center_shift': 'yc shift',
            'starting_level': 'start lvl',
            'num_levels': 'num lvls',
            'level_frequency': 'lvl freq',
            'pole_rotation': 'pole rot',
            'pole_long': 'pole lon',
            'pole_lat': 'pole lat',
            'fractal_points': 'fractal'
        }

        for name, min_val, max_val, res in params_config:
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=label_map[name], width=10, anchor='w')
            label.pack(side=tk.LEFT, padx=(0, 5))

            slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=self.params[name])
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

            entry = ttk.Entry(frame, width=8)
            entry.pack(side=tk.LEFT)

            if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
                entry.insert(0, str(int(self.params[name].get())))
            else:
                entry.insert(0, f"{self.params[name].get():.2f}")

            entry.bind('<Return>',
                       lambda e, n=name, ent=entry, mn=min_val, mx=max_val: self.on_entry_change(n, ent, mn, mx))
            entry.bind('<FocusOut>',
                       lambda e, n=name, ent=entry, mn=min_val, mx=max_val: self.on_entry_change(n, ent, mn, mx))

            self.params[name].entry = entry
            slider.config(command=lambda v, n=name, ent=entry: self.on_slider_change(n, ent))

        source_frame = ttk.LabelFrame(control_frame, text="Source", padding=5)
        source_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.source_canvas = tk.Canvas(source_frame, bg='#1f2121')
        self.source_canvas.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(main_frame, text="Result", padding=5)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(result_frame, bg='#1f2121')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def on_slider_change(self, name, entry):
        val = self.params[name].get()
        entry.delete(0, tk.END)
        if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
            entry.insert(0, str(int(val)))
        else:
            entry.insert(0, f"{val:.3f}")
        self.schedule_update(preview=True)

    def on_entry_change(self, name, entry, min_val, max_val):
        try:
            if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
                val = int(float(entry.get()))
            else:
                val = float(entry.get())

            val = max(min_val, min(max_val, val))
            self.params[name].set(val)

            entry.delete(0, tk.END)
            if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
                entry.insert(0, str(int(val)))
            else:
                entry.insert(0, f"{val:.3f}")

            self.schedule_update(preview=False)

        except ValueError:
            entry.delete(0, tk.END)
            if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
                entry.insert(0, str(int(self.params[name].get())))
            else:
                entry.insert(0, f"{self.params[name].get():.3f}")

    def schedule_update(self, preview=True):
        if self.update_pending:
            self.root.after_cancel(self.update_pending)

        if preview:
            self.update_pending = self.root.after(0, lambda: self.update_transform(preview=True))
        else:
            self.update_pending = self.root.after(300, lambda: self.update_transform(preview=False))

    def update_source_canvas(self):
        if self.transformer is None:
            return

        max_w = self.source_canvas.winfo_width()
        max_h = self.source_canvas.winfo_height()

        if max_w <= 1 or max_h <= 1:
            return

        source_img = self.transformer.source_img.copy()
        img_w, img_h = source_img.size
        scale = min(max_w / img_w, max_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        source_img = source_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.source_photo = ImageTk.PhotoImage(source_img)
        self.source_canvas.delete("all")
        self.source_canvas.create_image(max_w // 2, max_h // 2, image=self.source_photo)

    def update_transform(self, preview=True):
        if self.transformer is None or self.is_updating:
            return

        if not preview and self.full_render_thread and self.full_render_thread.is_alive():
            self.cancel_full_render = True
            return

        self.is_updating = True

        max_w = self.canvas.winfo_width()
        max_h = self.canvas.winfo_height()

        if max_w <= 1 or max_h <= 1:
            self.is_updating = False
            return

        final_img_w = self.transformer.width
        final_img_h = self.transformer.height
        display_scale = min(1.0, max_w / final_img_w, max_h / final_img_h)
        display_w = int(final_img_w * display_scale)
        display_h = int(final_img_h * display_scale)


        kwargs = {
            'inner_radius': self.params['inner_radius'].get(),
            'outer_radius': self.params['outer_radius'].get(),
            'periodicity': self.params['periodicity'].get(),
            'strands': self.params['strands'].get(),
            'zoom': self.params['zoom'].get(),
            'rotate': self.params['rotate'].get(),
            'x_shift': self.params['x_shift'].get(),
            'y_shift': self.params['y_shift'].get(),
            'x_center_shift': self.params['x_center_shift'].get(),
            'y_center_shift': self.params['y_center_shift'].get(),
            'starting_level': self.params['starting_level'].get(),
            'num_levels': self.params['num_levels'].get(),
            'level_frequency': self.params['level_frequency'].get(),
            'show_both_poles': self.show_both_poles.get(),
            'pole_rotation': self.params['pole_rotation'].get(),
            'pole_long': self.params['pole_long'].get(),
            'pole_lat': self.params['pole_lat'].get(),
            'tile_poles': self.tile_poles.get(),
            'hyper_droste': self.hyper_droste.get(),
            'fractal_points': self.params['fractal_points'].get(),
            'auto_periodicity': self.auto_periodicity.get(),
            'mirror_effect': self.mirror_effect.get(),
            'no_transparency': self.no_transparency.get(),
            'external_transparency': self.external_transparency.get(),
            'untwist': self.untwist.get(),
            'do_not_flatten_transparency': self.do_not_flatten_transparency.get(),
            'show_grid': self.show_grid.get(),
            'preview': preview
        }

        if preview:
            result = self.transformer.transform(**kwargs)

            self.display_result(result, display_w, display_h, preview)
            self.is_updating = False

            if self.update_pending:
                self.root.after_cancel(self.update_pending)
            self.update_pending = self.root.after(500, lambda: self.update_transform(preview=False))
        else:
            self.is_updating = False
            self.cancel_full_render = False
            self.full_render_thread = threading.Thread(
                target=self.render_full_in_background,
                args=(kwargs, display_w, display_h))
            self.full_render_thread.start()

    def render_full_in_background(self, kwargs, display_w, display_h):
        if self.cancel_full_render:
            return

        result = self.transformer.transform(**kwargs)

        if self.cancel_full_render:
            return

        self.last_full_result = result.copy()
        self.root.after(0, lambda: self.display_result(result, display_w, display_h, False))

    def display_result(self, result, display_w, display_h, preview):
        resample = Image.Resampling.NEAREST if preview else Image.Resampling.LANCZOS
        if result.size != (display_w, display_h):
            result = result.resize((display_w, display_h), resample)

        self.photo = ImageTk.PhotoImage(result)
        max_w = self.canvas.winfo_width()
        max_h = self.canvas.winfo_height()
        self.canvas.delete("all")
        self.canvas.create_image(max_w // 2, max_h // 2, image=self.photo)

    def get_config(self):
        return {
            'source_image': self.source_image_path,
            'params': {k: v.get() for k, v in self.params.items()},
            'show_both_poles': self.show_both_poles.get(),
            'tile_poles': self.tile_poles.get(),
            'hyper_droste': self.hyper_droste.get(),
            'auto_periodicity': self.auto_periodicity.get(),
            'mirror_effect': self.mirror_effect.get(),
            'untwist': self.untwist.get(),
            'no_transparency': self.no_transparency.get(),
            'external_transparency': self.external_transparency.get(),
            'do_not_flatten_transparency': self.do_not_flatten_transparency.get(),
            'show_grid': self.show_grid.get()
        }

    def set_config(self, config):
        params = config.get('params', {})
        for name, var in self.params.items():
            if name in params:
                var.set(params[name])
                if name in ['strands', 'starting_level', 'num_levels', 'level_frequency', 'fractal_points']:
                    var.entry.delete(0, tk.END)
                    var.entry.insert(0, str(int(params[name])))
                else:
                    var.entry.delete(0, tk.END)
                    var.entry.insert(0, f"{params[name]:.2f}")

        self.show_both_poles.set(config.get('show_both_poles', False))
        self.tile_poles.set(config.get('tile_poles', False))
        self.hyper_droste.set(config.get('hyper_droste', False))
        self.auto_periodicity.set(config.get('auto_periodicity', False))
        self.mirror_effect.set(config.get('mirror_effect', False))
        self.untwist.set(config.get('untwist', False))
        self.no_transparency.set(config.get('no_transparency', False))
        self.external_transparency.set(config.get('external_transparency', False))
        self.do_not_flatten_transparency.set(config.get('do_not_flatten_transparency', False))
        self.show_grid.set(config.get('show_grid', False))

    def save_last_image_config(self):
        if self.source_image_path:
            config = {'last_image': self.source_image_path}
            try:
                with open(self.CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                print(f"Failed to save config: {e}")

    def load_last_image(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                last_image = config.get('last_image')
                if last_image and os.path.exists(last_image):
                    self.source_image_path = last_image
                    self.transformer = DrosteTransform(last_image)
                    self.root.after(100, self.update_source_canvas)
                    self.root.after(150, lambda: self.update_transform(preview=False))
            except Exception as e:
                print(f"Failed to load last image: {e}")

    def load_image(self):
        initial_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])

        if filepath:
            self.source_image_path = filepath
            self.transformer = DrosteTransform(filepath)
            self.update_source_canvas()
            self.update_transform(preview=False)
            self.save_last_image_config()

    def load_config(self):
        initial_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json")])

        if filepath:
            with open(filepath, 'r') as f:
                config = json.load(f)

            source_path = config.get('source_image')
            if source_path and os.path.exists(source_path):
                self.source_image_path = source_path
                self.transformer = DrosteTransform(source_path)
                self.set_config(config)
                self.update_source_canvas()
                self.update_transform(preview=False)
                self.save_last_image_config()

    def save_image(self):
        if self.last_full_result is None:
            return

        initial_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])

        if filepath:
            self.last_full_result.save(filepath)

            base_path = os.path.splitext(filepath)[0]
            config_path = base_path + '_droste.json'
            config = self.get_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)


if __name__ == '__main__':
    root = tk.Tk()
    app = DrosteGUI(root)
    root.mainloop()
