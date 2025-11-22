#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import os
import time
import json


class MapsTransform:
    def __init__(self, image_path):
        self.source_img = Image.open(image_path).convert('RGB')
        self.width, self.height = self.source_img.size
        self.img_array = np.array(self.source_img).astype(np.float32) / 255.0

    def transform(self, centerx=0.0, centery=0.0, cx=0.20, cy=-0.20,
                  inzoom=1.0, iter_count=3, revzoom=1.0, posx=0.0, posy=0.0,
                  power=2, logistic=True, newton=False, sine=False, tricorn=False, burning_ship=False, exponential=False, trig=False,
                  boundary_mode='clip', back_color=(255, 0, 0), target_size=None):

        if target_size:
            height, width = target_size
        else:
            height, width = self.height, self.width

        y_coords = np.linspace(-1, 1, height)
        x_coords = np.linspace(-1, 1, width)
        xv, yv = np.meshgrid(x_coords, y_coords)

        rip = xv + 1j * yv
        rip = rip + (posx + 1j * posy)
        rip = rip * revzoom

        p = cx + 1j * cy

        with np.errstate(over='ignore', invalid='ignore'):
            for j in range(iter_count):
                if logistic:
                    rip = np.power(rip, power) + p
                if newton:
                    rip = rip - (np.power(rip, power) + p) / (power * np.power(rip, power - 1) + 1e-10)
                if sine:
                    rip = np.sin(np.power(rip, power)) + p
                if tricorn:
                    rip = np.conj(np.power(rip, power)) + p
                if burning_ship:
                    rip = np.power(np.abs(rip.real) + 1j * np.abs(rip.imag), power) + p
                if exponential:
                    rip =  np.power(power,rip)+p
                if trig:
                    rip =  np.sin(np.power( power, rip))+p

                rip = rip * inzoom

            rip = np.nan_to_num(rip, nan=0.0, posinf=0.0, neginf=0.0)

        sa_x = rip.real + centerx
        sa_y = rip.imag + centery

        sample_x = ((sa_x + 1) / 2) * (self.width - 1)
        sample_y = ((sa_y + 1) / 2) * (self.height - 1)

        sample_x = np.nan_to_num(sample_x, nan=0.0, posinf=0.0, neginf=0.0)
        sample_y = np.nan_to_num(sample_y, nan=0.0, posinf=0.0, neginf=0.0)

        if boundary_mode == 'torus':
            sample_x = sample_x % self.width
            sample_y = sample_y % self.height
        elif boundary_mode == 'reflect':
            period_x = 2 * self.width
            period_y = 2 * self.height

            sample_x = sample_x % period_x
            sample_y = sample_y % period_y

            sample_x = np.where(sample_x >= self.width, period_x - sample_x - 1, sample_x)
            sample_y = np.where(sample_y >= self.height, period_y - sample_y - 1, sample_y)
        elif boundary_mode == 'infinite':
            w = self.width - 1
            h = self.height - 1

            mask_x_high = sample_x > w
            mask_x_low = sample_x < 0
            mask_y_high = sample_y > h
            mask_y_low = sample_y < 0

            dist_x_high = sample_x - w
            sample_x = np.where(mask_x_high, w * (w + 1) / (w + dist_x_high), sample_x)

            dist_x_low = -sample_x
            sample_x = np.where(mask_x_low, w - w * (w + 1) / (w + dist_x_low), sample_x)

            dist_y_high = sample_y - h
            sample_y = np.where(mask_y_high, h * (h + 1) / (h + dist_y_high), sample_y)

            dist_y_low = -sample_y
            sample_y = np.where(mask_y_low, h - h * (h + 1) / (h + dist_y_low), sample_y)

            sample_x = np.clip(sample_x, 0, w)
            sample_y = np.clip(sample_y, 0, h)
        else:  # clip
            sample_x = np.clip(sample_x, 0, self.width - 1)
            sample_y = np.clip(sample_y, 0, self.height - 1)

        x0 = sample_x.astype(int)
        y0 = sample_y.astype(int)
        x1 = np.minimum(x0 + 1, self.width - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)

        wx = sample_x - x0
        wy = sample_y - y0

        result = np.zeros((height, width, 3), dtype=np.float32)

        for c in range(3):
            c00 = self.img_array[y0, x0, c]
            c01 = self.img_array[y0, x1, c]
            c10 = self.img_array[y1, x0, c]
            c11 = self.img_array[y1, x1, c]

            result[:, :, c] = (c00 * (1 - wx) * (1 - wy) +
                               c01 * wx * (1 - wy) +
                               c10 * (1 - wx) * wy +
                               c11 * wx * wy)

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


class MapsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Iterative Maps ")
        self.root.geometry("2000x1400")

        self.transformer = None
        self.photo = None
        self.source_photo = None
        self.last_full_result = None
        self.source_image_path = None

        self.mode = tk.StringVar(value='logistic')

        self.params = {
            'centerx': tk.DoubleVar(value=0.0),
            'centery': tk.DoubleVar(value=0.0),
            'cx': tk.DoubleVar(value=0.20),
            'cy': tk.DoubleVar(value=-0.20),
            'inzoom': tk.DoubleVar(value=1.0),
            'revzoom': tk.DoubleVar(value=1.0),
            'posx': tk.DoubleVar(value=0.0),
            'posy': tk.DoubleVar(value=0.0),
            'iter': tk.IntVar(value=3),
            'power': tk.DoubleVar(value=2.0)
        }

        self.boundary_mode = tk.StringVar(value='clip')

        self.update_pending = None
        self.is_updating = False
        self.preview_max_size = 128

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame, width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.TOP, fill=tk.X, pady=2)

        mode_frame = ttk.LabelFrame(control_frame, text="Mode", padding=5)
        mode_frame.pack(pady=10, fill=tk.X)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode,
                                  values=['logistic', 'newton', 'sine', 'tricorn', 'burning_ship', 'exponential', 'trig'],
                                  state='readonly', width=15)
        mode_combo.pack(pady=5, fill=tk.X)
        mode_combo.bind('<<ComboboxSelected>>', lambda e: self.schedule_update())

        boundary_frame = ttk.LabelFrame(control_frame, text="Boundary", padding=5)
        boundary_frame.pack(pady=10, fill=tk.X)
        ttk.Label(boundary_frame, text="Mode:").pack(anchor=tk.W)
        boundary_combo = ttk.Combobox(boundary_frame, textvariable=self.boundary_mode,
                                      values=['clip', 'torus', 'reflect', 'infinite'],
                                      state='readonly', width=15)
        boundary_combo.pack(pady=5, fill=tk.X)
        boundary_combo.bind('<<ComboboxSelected>>', lambda e: self.schedule_update())

        params_config = [
            ('centerx', -1.5, 1.5, 0.001),
            ('centery', -1.5, 1.5, 0.001),
            ('cx', -3.0, 3.0, 0.0001),
            ('cy', -3.0, 3.0, 0.0001),
            ('inzoom', -3.0, 3.0, 0.01),
            ('revzoom', -10.0, 10.0, 0.01),
            ('posx', -3.0, 3.0, 0.001),
            ('posy', -3.0, 3.0, 0.001),
            ('iter', 0, 20, 1),
            ('power', 0.0, 20, 0.01)
        ]

        for name, min_val, max_val, res in params_config:
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=3)

            label = ttk.Label(frame, text=f"{name}:")
            label.pack(side=tk.TOP, anchor=tk.W)

            slider_frame = ttk.Frame(frame)
            slider_frame.pack(fill=tk.X)

            slider = ttk.Scale(slider_frame, from_=min_val, to=max_val,
                               variable=self.params[name])
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            entry = ttk.Entry(slider_frame, width=8)
            entry.pack(side=tk.LEFT, padx=(5, 0))

            if name in ['iter']:
                entry.insert(0, str(int(self.params[name].get())))
            else:
                entry.insert(0, f"{self.params[name].get():.2f}")

            entry.bind('<Return>', lambda e, n=name, ent=entry, mn=min_val, mx=max_val:
            self.on_entry_change(n, ent, mn, mx))
            entry.bind('<FocusOut>', lambda e, n=name, ent=entry, mn=min_val, mx=max_val:
            self.on_entry_change(n, ent, mn, mx))

            self.params[name].entry = entry
            slider.config(command=lambda v, n=name, ent=entry: self.on_slider_change(n, ent))

        source_frame = ttk.LabelFrame(control_frame, text="Source (click: center, wheel: inzoom)", padding=5)
        source_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.source_canvas = tk.Canvas(source_frame, bg='#1f2121')
        self.source_canvas.pack(fill=tk.BOTH, expand=True)
        self.source_canvas.bind('<Button-1>', self.on_source_click)
        self.source_canvas.bind('<MouseWheel>', self.on_source_wheel)
        self.source_canvas.bind('<Button-4>', self.on_source_wheel)
        self.source_canvas.bind('<Button-5>', self.on_source_wheel)

        result_frame = ttk.LabelFrame(main_frame, text="Result (click: position, wheel: revzoom)", padding=5)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(result_frame, bg='#1f2121')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.on_result_click)
        self.canvas.bind('<MouseWheel>', self.on_result_wheel)
        self.canvas.bind('<Button-4>', self.on_result_wheel)
        self.canvas.bind('<Button-5>', self.on_result_wheel)

    def on_source_click(self, event):
        if self.transformer is None:
            return

        w = self.source_canvas.winfo_width()
        h = self.source_canvas.winfo_height()

        norm_x = (event.x / w) * 2 - 1
        norm_y = (event.y / h) * 2 - 1

        self.params['centerx'].set(norm_x)
        self.params['centery'].set(norm_y)
        self.params['centerx'].entry.delete(0, tk.END)
        self.params['centerx'].entry.insert(0, f"{norm_x:.3f}")
        self.params['centery'].entry.delete(0, tk.END)
        self.params['centery'].entry.insert(0, f"{norm_y:.3f}")

        self.schedule_update(preview=True)

    def on_source_wheel(self, event):
        if self.transformer is None:
            return

        delta = 0
        if event.num == 4 or event.delta > 0:
            delta = 0.1
        elif event.num == 5 or event.delta < 0:
            delta = -0.1

        current = self.params['inzoom'].get()
        new_val = max(-3.0, min(3.0, current + delta))
        self.params['inzoom'].set(new_val)
        self.params['inzoom'].entry.delete(0, tk.END)
        self.params['inzoom'].entry.insert(0, f"{new_val:.3f}")

        self.schedule_update(preview=True)

    def on_result_click(self, event):
        if self.transformer is None:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        norm_x = -((event.x / w) * 2 - 1)
        norm_y = -((event.y / h) * 2 - 1)

        self.params['posx'].set(norm_x)
        self.params['posy'].set(norm_y)
        self.params['posx'].entry.delete(0, tk.END)
        self.params['posx'].entry.insert(0, f"{norm_x:.3f}")
        self.params['posy'].entry.delete(0, tk.END)
        self.params['posy'].entry.insert(0, f"{norm_y:.3f}")

        self.schedule_update(preview=True)

    def on_result_wheel(self, event):
        if self.transformer is None:
            return

        delta = 0
        if event.num == 4 or event.delta > 0:
            delta = 0.1
        elif event.num == 5 or event.delta < 0:
            delta = -0.1

        current = self.params['revzoom'].get()
        new_val = max(-10.0, min(10.0, current + delta))
        self.params['revzoom'].set(new_val)
        self.params['revzoom'].entry.delete(0, tk.END)
        self.params['revzoom'].entry.insert(0, f"{new_val:.3f}")

        self.schedule_update(preview=True)

    def on_slider_change(self, name, entry):
        val = self.params[name].get()
        entry.delete(0, tk.END)
        if name in ['iter']:
            entry.insert(0, str(int(val)))
        else:
            entry.insert(0, f"{val:.3f}")
        self.schedule_update(preview=True)

    def on_entry_change(self, name, entry, min_val, max_val):
        try:
            if name in ['iter']:
                val = int(float(entry.get()))
            else:
                val = float(entry.get())
            val = max(min_val, min(max_val, val))
            self.params[name].set(val)
            entry.delete(0, tk.END)
            if name in ['iter']:
                entry.insert(0, str(int(val)))
            else:
                entry.insert(0, f"{val:.3f}")
            self.schedule_update(preview=False)
        except ValueError:
            entry.delete(0, tk.END)
            if name in ['iter']:
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

        self.is_updating = True
        t_start = time.time()

        mode_flags = {
            'logistic': self.mode.get() == 'logistic',
            'newton': self.mode.get() == 'newton',
            'sine': self.mode.get() == 'sine',
            'tricorn': self.mode.get() == 'tricorn',
            'burning_ship': self.mode.get() == 'burning_ship',
            'exponential': self.mode.get() == 'exponential',
            'trig': self.mode.get() == 'trig'
        }

        max_w = self.canvas.winfo_width()
        max_h = self.canvas.winfo_height()

        if max_w <= 1 or max_h <= 1:
            self.is_updating = False
            return

        # Calculate final image display dimensions
        final_img_w = self.transformer.width
        final_img_h = self.transformer.height
        display_scale = min(1.0, max_w / final_img_w, max_h / final_img_h)
        display_w = int(final_img_w * display_scale)
        display_h = int(final_img_h * display_scale)

        # Determine render target
        if preview:
            src_aspect = final_img_h / final_img_w
            target_w = self.preview_max_size
            target_h = int(target_w * src_aspect)
            target_size = (target_h, target_w)
        else:
            target_size = None

        result = self.transformer.transform(
            centerx=self.params['centerx'].get(),
            centery=self.params['centery'].get(),
            cx=self.params['cx'].get(),
            cy=self.params['cy'].get(),
            inzoom=self.params['inzoom'].get(),
            iter_count=self.params['iter'].get(),
            revzoom=self.params['revzoom'].get(),
            posx=self.params['posx'].get(),
            posy=self.params['posy'].get(),
            power=self.params['power'].get(),
            boundary_mode=self.boundary_mode.get(),
            target_size=target_size,
            **mode_flags
        )

        if not preview:
            self.last_full_result = result.copy()

        # Scale to display dimensions
        resample = Image.Resampling.NEAREST if preview else Image.Resampling.LANCZOS
        if result.size != (display_w, display_h):
            result = result.resize((display_w, display_h), resample)

        t_scale = time.time()

        self.photo = ImageTk.PhotoImage(result)
        self.canvas.delete("all")
        self.canvas.create_image(max_w // 2, max_h // 2, image=self.photo)

        # t_display = time.time()

        # render_ms = (t_render - t_start) * 1000
        # scale_ms = (t_scale - t_render) * 1000
        # display_ms = (t_display - t_scale) * 1000
        # total_ms = (t_display - t_start) * 1000
        #
        # mode_str = "PREVIEW" if preview else "FULL"
        # size_str = f"{result.size[0]}x{result.size[1]}"
        # print(
        #     f"[{mode_str:7s}] {size_str:10s} | render: {render_ms:6.2f}ms | scale: {scale_ms:5.2f}ms | display: {display_ms:5.2f}ms | total: {total_ms:6.2f}ms")

        self.is_updating = False

        if preview:
            if self.update_pending:
                self.root.after_cancel(self.update_pending)
            self.update_pending = self.root.after(500, lambda: self.update_transform(preview=False))

    def get_config(self):
        return {
            'source_image': self.source_image_path,
            'mode': self.mode.get(),
            'boundary_mode': self.boundary_mode.get(),
            'params': {
                'centerx': self.params['centerx'].get(),
                'centery': self.params['centery'].get(),
                'cx': self.params['cx'].get(),
                'cy': self.params['cy'].get(),
                'inzoom': self.params['inzoom'].get(),
                'revzoom': self.params['revzoom'].get(),
                'posx': self.params['posx'].get(),
                'posy': self.params['posy'].get(),
                'iter': self.params['iter'].get(),
                'power': self.params['power'].get()
            }
        }

    def set_config(self, config):
        self.mode.set(config.get('mode', 'logistic'))
        self.boundary_mode.set(config.get('boundary_mode', 'clip'))

        params = config.get('params', {})
        for name, var in self.params.items():
            if name in params:
                var.set(params[name])
                if name in ['iter']:
                    var.entry.delete(0, tk.END)
                    var.entry.insert(0, str(int(params[name])))
                else:
                    var.entry.delete(0, tk.END)
                    var.entry.insert(0, f"{params[name]:.2f}")

    def load_image(self):
        initial_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if filepath:
            self.source_image_path = filepath
            self.transformer = MapsTransform(filepath)
            self.update_source_canvas()
            self.update_transform(preview=False)

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
                self.transformer = MapsTransform(source_path)
                self.set_config(config)
                self.update_source_canvas()
                self.update_transform(preview=False)

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
            config_path = base_path + '.json'

            config = self.get_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)


if __name__ == '__main__':
    root = tk.Tk()
    app = MapsGUI(root)
    root.mainloop()
