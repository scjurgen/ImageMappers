import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


class DrosteTransform:
    def __init__(self, image_path, preview_size=128):
        self.source_img = Image.open(image_path).convert('RGBA')
        self.width, self.height = self.source_img.size
        self.img_array = np.array(self.source_img).astype(np.float32) / 255.0

        # Generate downsampled preview
        max_dim = max(self.width, self.height)
        scale = preview_size / max_dim
        self.preview_width = int(self.width * scale)
        self.preview_height = int(self.height * scale)

        preview_img = self.source_img.resize((self.preview_width, self.preview_height), Image.LANCZOS)
        self.img_preview_array = np.array(preview_img).astype(np.float32) / 255.0

    def transform(self, inner_radius=25, outer_radius=100, periodicity=1.0, strands=1,
                  zoom=1, rotate=0, x_shift=0, y_shift=0,
                  x_center_shift=0, y_center_shift=0,
                  starting_level=1, num_levels=10, level_frequency=1,
                  show_both_poles=False, pole_rotation=90, pole_long=0, pole_lat=0,
                  tile_poles=False, hyper_droste=False, fractal_points=1,
                  auto_periodicity=False, mirror_effect=False, untwist=False,
                  no_transparency=False, external_transparency=False,
                  do_not_flatten_transparency=False, show_grid=False,
                  show_frame=False, preview=False):
        try:
            if preview:
                height, width = self.preview_height, self.preview_width
                img_array = self.img_preview_array
                img_width, img_height = self.preview_width, self.preview_height
            else:
                height, width = self.height, self.width
                img_array = self.img_array
                img_width, img_height = self.width, self.height

            W, H = width, height
            X = max(width, height) / 2.0
            Y = max(width, height) / 2.0

            r1 = inner_radius / 100.0
            r2 = outer_radius / 100.0
            p1 = periodicity
            p2 = strands

            x_center_shift_scaled = x_center_shift / 100.0
            y_center_shift_scaled = y_center_shift / 100.0
            x_shift_scaled = (x_shift * W / X) / 100.0
            y_shift_scaled = (y_shift * H / Y) / 100.0

            tile_based_on_transparency = not no_transparency
            transparent_points_in = not external_transparency
            retwist = not untwist

            if auto_periodicity:
                p1 = p2 / 2.0 * (1 + np.sqrt(1 - (np.log(r2 / r1) / np.pi) ** 2))

            if p1 > 0:
                rotate_rad = -(np.pi / 180) * rotate
            else:
                rotate_rad = (np.pi / 180) * rotate

            zoom_factor = (zoom + inner_radius - 1) / 100.0
            epsilon = 0.01

            if retwist:
                xbounds = [-r2, r2]
                ybounds = [-r2, r2]
            else:
                ybounds = [0, 2.1 * np.pi]
                xbounds = [-np.log(r2 / r1), np.log(r2 / r1)]

            min_dimension = min(W, H)
            xymiddle = np.array([(xbounds[0] + xbounds[1]) / 2.0,
                                 (ybounds[0] + ybounds[1]) / 2.0])

            xyrange = np.array([xbounds[1] - xbounds[0], ybounds[1] - ybounds[0]])
            aspect_ratio = W / H
            xyrange[0] = xyrange[1] * aspect_ratio
            xbounds = [xymiddle[0] - 0.5 * xyrange[0], xymiddle[0] + 0.5 * xyrange[0]]

            x_pixel = np.arange(width) - W / 2.0
            y_pixel = np.arange(height) - H / 2.0
            xv, yv = np.meshgrid(x_pixel, y_pixel)

            z_real = xbounds[0] + (xbounds[1] - xbounds[0]) * (xv + W / 2) / W
            z_imag = ybounds[0] + (ybounds[1] - ybounds[0]) * (yv + H / 2) / H
            z = z_real + 1j * z_imag

            if retwist:
                z_initial = z.copy()
                z = z - (x_shift_scaled + 1j * y_shift_scaled)
                z = (xymiddle[0] + 1j * xymiddle[1]) + \
                    (z - (xymiddle[0] + 1j * xymiddle[1])) / zoom_factor * np.exp(-1j * rotate_rad)
            else:
                z_initial = r1 * np.exp(z)
                z_initial = z_initial * zoom * np.exp(1j * rotate_rad)

            if show_both_poles:
                theta = (np.pi / 180) * pole_rotation
                xx = z.real
                yy = z.imag
                div = 0.5 * (1 + xx ** 2 + yy ** 2 + ((1 - xx ** 2 - yy ** 2) * np.cos(theta)) - (2 * xx * np.sin(theta)))
                xx = xx * np.cos(theta) + (0.5 * (1 - xx ** 2 - yy ** 2) * np.sin(theta))
                z = (xx + 1j * yy) / div
            else:
                if hyper_droste:
                    z = np.sin(z)
                if tile_poles:
                    z = z ** fractal_points
                    z = np.tan(2 * z)

            p_lat = (pole_lat * W / X) / 100.0
            p_lon = (pole_long * W / X) / 100.0
            z = z + (p_lat + 1j * p_lon)

            if retwist:
                z2 = np.log(z / r1)
            else:
                z2 = z

            alpha = np.arctan(p2 / p1 * np.log(r2 / r1) / (2 * np.pi))
            f = np.cos(alpha)
            beta = f * np.exp(1j * alpha)

            if p2 > 0:
                angle = 2 * np.pi * p1
            else:
                angle = -2 * np.pi * p1

            if mirror_effect:
                angle = angle / strands

            z = p1 * z2 / beta
            rotatedscaledlogz = z.copy()
            logz = z2.copy()
            z = r1 * np.exp(z)

            if tile_based_on_transparency and starting_level > 0:
                if not transparent_points_in:
                    ratio = r2 / r1 * np.exp(1j * angle)
                else:
                    ratio = r1 / r2 * np.exp(-1j * angle)
                z = z * (ratio ** starting_level)

            color_so_far = np.zeros((height, width, 4), dtype=np.float32)
            alpha_remaining = np.ones((height, width), dtype=np.float32)

            def sample_image(z_sample):
                # MathMap uses xy:[-X..X, -Y..Y] with Y increasing upward
                # Convert to pixel coordinates [0..width-1, 0..height-1] with Y increasing downward
                ix = (z_sample.real + x_center_shift_scaled) * (min_dimension / 2.0) + img_width / 2.0
                iy = (-z_sample.imag - y_center_shift_scaled) * (min_dimension / 2.0) + img_height / 2.0

                valid_mask = (ix >= 0) & (ix < img_width - 1) & (iy >= 0) & (iy < img_height - 1)
                coords = np.array([iy, ix])

                color = np.zeros((height, width, 4), dtype=np.float32)
                for c in range(4):
                    sampled = map_coordinates(img_array[:, :, c], coords,
                                              order=1, mode='constant', cval=0)
                    color[:, :, c] = np.where(valid_mask, sampled, 0)
                return color

            color_out = sample_image(z)
            color_so_far = color_so_far + color_out * color_out[:, :, 3:4] * alpha_remaining[:, :, np.newaxis]
            alpha_remaining = alpha_remaining * (1 - color_out[:, :, 3])

            sign = np.zeros((height, width), dtype=np.float32)
            if tile_based_on_transparency:
                if transparent_points_in:
                    sign = np.where(alpha_remaining > epsilon, -1, 0)
                else:
                    sign = np.where(alpha_remaining > epsilon, 1, 0)
            else:
                radius = np.abs(z)
                sign = np.where(radius < r1, -1, sign)
                sign = np.where(radius > r2, 1, sign)

            ratio = np.where(sign < 0, r2 / r1 * np.exp(1j * angle),
                             np.where(sign > 0, r1 / r2 * np.exp(-1j * angle), 1))

            if level_frequency > 1:
                ratio = np.exp(np.log(ratio) * level_frequency)

            iteration = starting_level
            max_iteration = num_levels + starting_level - 1

            while np.any(sign != 0) and iteration < max_iteration:
                z = z * ratio
                rotatedscaledlogz = rotatedscaledlogz + (0 - 1j * sign * angle)

                sign_mask = (sign != 0)

                if tile_based_on_transparency:
                    color_out = sample_image(z)
                    color_so_far = color_so_far + color_out * color_out[:, :, 3:4] * alpha_remaining[:, :, np.newaxis]
                    alpha_remaining = alpha_remaining * (1 - color_out[:, :, 3])

                    sign = np.zeros((height, width), dtype=np.float32)
                    if transparent_points_in:
                        sign = np.where(alpha_remaining > epsilon, -1, 0)
                    else:
                        sign = np.where(alpha_remaining > epsilon, 1, 0)
                else:
                    radius = np.abs(z)
                    color_out = sample_image(z)
                    color_so_far[:, :, :3] = np.where(sign_mask[:, :, np.newaxis],
                                                      color_out[:, :, :3],
                                                      color_so_far[:, :, :3])

                    sign = np.zeros((height, width), dtype=np.float32)
                    sign = np.where(radius < r1, -1, sign)
                    sign = np.where(radius > r2, 1, sign)

                ratio = np.where(sign < 0, r2 / r1 * np.exp(1j * angle),
                                 np.where(sign > 0, r1 / r2 * np.exp(-1j * angle), 1))

                if level_frequency > 1:
                    ratio = np.exp(np.log(ratio) * level_frequency)

                iteration += 1

            color_out = color_so_far

            if show_grid:
                gridz_real = (logz.real + 10 * np.log(r2 / r1)) % np.log(r2 / r1)
                gridz_imag = (logz.imag + 10 * 2 * np.pi) % (2 * np.pi)

                green_mask = (gridz_real < epsilon) | (gridz_real > (np.log(r2 / r1) - epsilon)) | \
                             (gridz_imag < epsilon) | (gridz_imag > (2 * np.pi - epsilon))

                color_out = np.where(green_mask[:, :, np.newaxis],
                                     np.array([0, 1, 0, 1]), color_out)

                gridz_real = (rotatedscaledlogz.real + 10 * np.log(r2 / r1)) % np.log(r2 / r1)
                gridz_imag = (rotatedscaledlogz.imag + 10 * 2 * np.pi) % (2 * np.pi)

                blue_mask = (gridz_real < epsilon) | (gridz_real > (np.log(r2 / r1) - epsilon)) | \
                            (gridz_imag < epsilon) | (gridz_imag > (2 * np.pi - epsilon))

                color_out = np.where(blue_mask[:, :, np.newaxis],
                                     np.array([0, 0, 1, 1]), color_out)

            if show_frame:
                gridz_real = z_initial.real
                gridz_imag = z_initial.imag

                in_frame = (gridz_real < (aspect_ratio * r2)) & (gridz_real > -(aspect_ratio * r2)) & \
                           (gridz_imag < r2) & (gridz_imag > -r2)

                dx = np.minimum((aspect_ratio * r2) - gridz_real, gridz_real + (aspect_ratio * r2))
                dy = np.minimum(r2 - gridz_imag, gridz_imag + r2)

                white_frame = in_frame & ((dx < 4 * epsilon) | (dy < 4 * epsilon))
                black_frame = in_frame & ((dx < 2 * epsilon) | (dy < 2 * epsilon))

                color_out = np.where(white_frame[:, :, np.newaxis],
                                     np.array([1, 1, 1, 1]), color_out)
                color_out = np.where(black_frame[:, :, np.newaxis],
                                     np.array([0, 0, 0, 1]), color_out)
                color_out = np.where((~in_frame)[:, :, np.newaxis],
                                     color_out * np.array([0.75, 0.75, 0.75, 1]), color_out)

            if not do_not_flatten_transparency:
                color_out[:, :, 3] = 1.0

            output = np.clip(color_out * 255, 0, 255).astype(np.uint8)
            output = np.flipud(output)
            return Image.fromarray(output, mode='RGBA')
        except:
            print("pass")
            pass