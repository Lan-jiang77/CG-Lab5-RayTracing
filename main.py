import taichi as ti
import math

ti.init(arch=ti.cpu)

WIDTH = 960
HEIGHT = 540

PI = math.pi
EPS = 1e-4
INF = 1e10
MAX_SAMPLES = 8

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())
samples_per_pixel = ti.field(dtype=ti.i32, shape=())

MAT_DIFFUSE_RED = 1
MAT_MIRROR = 2
MAT_GROUND = 3


@ti.func
def normalize(v):
    return v / ti.sqrt(v.dot(v) + 1e-8)


@ti.func
def reflect(i, n):
    return i - 2.0 * i.dot(n) * n


@ti.func
def clamp01(x):
    return ti.min(ti.max(x, 0.0), 1.0)


@ti.func
def generate_ray(i, j, ox, oy):
    cam_pos = ti.Vector([0.0, 1.2, 5.5])
    look_at = ti.Vector([0.0, 0.0, 0.0])
    up = ti.Vector([0.0, 1.0, 0.0])

    forward = normalize(look_at - cam_pos)
    right = normalize(forward.cross(up))
    cam_up = normalize(right.cross(forward))

    fov = 55.0 * PI / 180.0
    aspect = WIDTH / HEIGHT

    u = ((ti.cast(i, ti.f32) + ox) / WIDTH - 0.5) * 2.0
    v = ((ti.cast(j, ti.f32) + oy) / HEIGHT - 0.5) * 2.0

    u *= aspect * ti.tan(fov * 0.5)
    v *= ti.tan(fov * 0.5)

    ray_dir = normalize(forward + u * right + v * cam_up)
    return cam_pos, ray_dir


@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    h = b * b - c

    t = INF

    if h > 0.0:
        h = ti.sqrt(h)
        t1 = -b - h
        t2 = -b + h

        if t1 > EPS:
            t = t1
        elif t2 > EPS:
            t = t2

    return t


@ti.func
def intersect_plane(ro, rd):
    t = INF

    if ti.abs(rd.y) > 1e-6:
        temp = (-1.0 - ro.y) / rd.y
        if temp > EPS:
            t = temp

    return t


@ti.func
def scene_intersect(ro, rd):
    hit_t = INF
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    mat_id = 0

    red_center = ti.Vector([-1.5, 0.0, 0.0])
    t_red = intersect_sphere(ro, rd, red_center, 1.0)

    if t_red < hit_t:
        hit_t = t_red
        hit_pos = ro + rd * hit_t
        hit_normal = normalize(hit_pos - red_center)
        mat_id = MAT_DIFFUSE_RED

    mirror_center = ti.Vector([1.5, 0.0, 0.0])
    t_mirror = intersect_sphere(ro, rd, mirror_center, 1.0)

    if t_mirror < hit_t:
        hit_t = t_mirror
        hit_pos = ro + rd * hit_t
        hit_normal = normalize(hit_pos - mirror_center)
        mat_id = MAT_MIRROR

    t_plane = intersect_plane(ro, rd)

    if t_plane < hit_t:
        hit_t = t_plane
        hit_pos = ro + rd * hit_t
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        mat_id = MAT_GROUND

    return hit_t, hit_pos, hit_normal, mat_id


@ti.func
def checker_color(p):
    x_id = ti.floor(p.x)
    z_id = ti.floor(p.z)

    checker = ti.cast(x_id + z_id, ti.i32) & 1

    color = ti.Vector([0.82, 0.82, 0.82])
    if checker == 1:
        color = ti.Vector([0.18, 0.18, 0.18])

    return color


@ti.func
def shadow_test(p, n):
    lp = light_pos[None]
    to_light = lp - p
    dist_to_light = ti.sqrt(to_light.dot(to_light))
    light_dir = to_light / dist_to_light

    shadow_origin = p + n * EPS
    t, hp, hn, mid = scene_intersect(shadow_origin, light_dir)

    in_shadow = False
    if t < dist_to_light:
        in_shadow = True

    return in_shadow, light_dir, dist_to_light


@ti.func
def shade_diffuse(p, n, base_color):
    ambient = 0.12
    color = base_color * ambient

    in_shadow, light_dir, light_dist = shadow_test(p, n)

    if not in_shadow:
        ndotl = ti.max(n.dot(light_dir), 0.0)
        attenuation = 1.0 / (0.10 * light_dist * light_dist + 1.0)
        diffuse = base_color * ndotl * attenuation * 4.0
        color += diffuse

    return color


@ti.func
def trace(ro, rd):
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    cur_ro = ro
    cur_rd = rd

    for bounce in range(8):
        if bounce < max_bounces[None]:
            t, p, n, mat_id = scene_intersect(cur_ro, cur_rd)

            if mat_id == 0:
                sky = ti.Vector([0.05, 0.12, 0.15])
                final_color += throughput * sky
                break

            if mat_id == MAT_DIFFUSE_RED:
                base = ti.Vector([0.85, 0.12, 0.08])
                c = shade_diffuse(p, n, base)
                final_color += throughput * c
                break

            elif mat_id == MAT_GROUND:
                base = checker_color(p)
                c = shade_diffuse(p, n, base)
                final_color += throughput * c
                break

            elif mat_id == MAT_MIRROR:
                reflectivity = 0.85
                throughput *= ti.Vector([reflectivity, reflectivity, reflectivity])

                cur_ro = p + n * EPS
                cur_rd = normalize(reflect(cur_rd, n))

    return final_color


@ti.kernel
def render():
    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])
        spp = samples_per_pixel[None]

        for s in range(MAX_SAMPLES):
            if s < spp:
                ox = ti.random(ti.f32)
                oy = ti.random(ti.f32)

                ro, rd = generate_ray(i, j, ox, oy)
                color += trace(ro, rd)

        color /= ti.cast(spp, ti.f32)

        color = ti.Vector([
            ti.sqrt(clamp01(color.x)),
            ti.sqrt(clamp01(color.y)),
            ti.sqrt(clamp01(color.z))
        ])

        pixels[i, j] = color


def main():
    light_x = 2.5
    light_y = 4.0
    light_z = 3.5
    bounce_count = 3
    spp = 4

    light_pos[None] = ti.Vector([light_x, light_y, light_z])
    max_bounces[None] = bounce_count
    samples_per_pixel[None] = spp

    window = ti.ui.Window("Ray Tracing Demo", (WIDTH, HEIGHT), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Controls", 0.70, 0.03, 0.27, 0.32):
            gui.text("Whitted-Style Ray Tracing")

            light_x = gui.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -5.0, 5.0)

            bounce_count = gui.slider_int("Max Bounces", bounce_count, 1, 5)
            spp = gui.slider_int("Samples", spp, 1, MAX_SAMPLES)

        light_pos[None] = ti.Vector([light_x, light_y, light_z])
        max_bounces[None] = bounce_count
        samples_per_pixel[None] = spp

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()