import taichi as ti
import math

# ============================================================
# 第五次实验：Whitted-Style Ray Tracing 光线追踪
#
# 已完成内容：
# 任务 1：搭建包含平面的三维场景
# 任务 2：实现基于迭代的光线弹射
# 任务 3：实现硬阴影，并解决 Shadow Acne
# 任务 4：完成 UI 交互面板
# 选做：MSAA 抗锯齿
# ============================================================

# 为了保证在普通电脑上也能运行，这里使用 CPU。
# 如果电脑 GPU 支持 Taichi，也可以改成 ti.gpu。
ti.init(arch=ti.cpu)

# ============================================================
# 一、全局参数设置
# ============================================================

WIDTH = 960
HEIGHT = 540

PI = math.pi
EPS = 1e-4          # 用于避免 Shadow Acne 的微小偏移量
INF = 1e10          # 表示无穷远
MAX_SAMPLES = 8     # MSAA 最大采样次数

# 图像像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# UI 控制参数
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())
samples_per_pixel = ti.field(dtype=ti.i32, shape=())

# 材质编号
MAT_DIFFUSE_RED = 1
MAT_MIRROR = 2
MAT_GROUND = 3


# ============================================================
# 二、基础数学工具函数
# ============================================================

@ti.func
def normalize(v):
    """
    向量归一化。
    加上一个很小的数，避免除以 0。
    """
    return v / ti.sqrt(v.dot(v) + 1e-8)


@ti.func
def reflect(i, n):
    """
    反射方向计算。

    反射公式：
        R = I - 2 * dot(I, N) * N

    其中：
        I 为入射光线方向
        N 为表面法向量
        R 为反射光线方向
    """
    return i - 2.0 * i.dot(n) * n


@ti.func
def clamp01(x):
    """
    将颜色限制到 [0, 1] 区间。
    """
    return ti.min(ti.max(x, 0.0), 1.0)


# ============================================================
# 三、相机与主光线生成
# 对应 PPT：Primary Ray 从摄像机发出
# ============================================================

@ti.func
def generate_ray(i, j, ox, oy):
    """
    根据像素坐标生成一条主光线。

    参数：
        i, j：当前像素坐标
        ox, oy：像素内部随机偏移，用于 MSAA 抗锯齿

    返回：
        cam_pos：光线起点
        ray_dir：光线方向
    """
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


# ============================================================
# 四、任务 1：搭建三维场景
# 场景包含：
# 1. 无限大棋盘平面
# 2. 红色漫反射球
# 3. 银色镜面球
# ============================================================

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """
    光线与球体求交。

    ro：ray origin，光线起点
    rd：ray direction，光线方向
    center：球心
    radius：半径

    返回：
        t：交点距离。
        如果没有相交，则返回 INF。
    """
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
    """
    光线与无限大平面求交。

    本实验中平面固定为：
        y = -1.0

    法向量为：
        (0, 1, 0)
    """
    t = INF

    if ti.abs(rd.y) > 1e-6:
        temp = (-1.0 - ro.y) / rd.y

        if temp > EPS:
            t = temp

    return t


@ti.func
def scene_intersect(ro, rd):
    """
    场景求交函数。

    依次检测：
        1. 红色漫反射球
        2. 银色镜面球
        3. 棋盘平面

    返回：
        hit_t：最近交点距离
        hit_pos：交点位置
        hit_normal：交点法线
        mat_id：材质编号
    """
    hit_t = INF
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    mat_id = 0

    # 左侧红色漫反射球
    red_center = ti.Vector([-1.5, 0.0, 0.0])
    t_red = intersect_sphere(ro, rd, red_center, 1.0)

    if t_red < hit_t:
        hit_t = t_red
        hit_pos = ro + rd * hit_t
        hit_normal = normalize(hit_pos - red_center)
        mat_id = MAT_DIFFUSE_RED

    # 右侧银色镜面球
    mirror_center = ti.Vector([1.5, 0.0, 0.0])
    t_mirror = intersect_sphere(ro, rd, mirror_center, 1.0)

    if t_mirror < hit_t:
        hit_t = t_mirror
        hit_pos = ro + rd * hit_t
        hit_normal = normalize(hit_pos - mirror_center)
        mat_id = MAT_MIRROR

    # 无限大棋盘平面
    t_plane = intersect_plane(ro, rd)

    if t_plane < hit_t:
        hit_t = t_plane
        hit_pos = ro + rd * hit_t
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        mat_id = MAT_GROUND

    return hit_t, hit_pos, hit_normal, mat_id


@ti.func
def checker_color(p):
    """
    根据交点的 x 和 z 坐标生成黑白棋盘格纹理。
    """
    x_id = ti.floor(p.x)
    z_id = ti.floor(p.z)

    checker = ti.cast(x_id + z_id, ti.i32) & 1

    color = ti.Vector([0.82, 0.82, 0.82])

    if checker == 1:
        color = ti.Vector([0.18, 0.18, 0.18])

    return color


# ============================================================
# 五、任务 3：硬阴影与 Shadow Acne 修复
# ============================================================

@ti.func
def shadow_test(p, n):
    """
    阴影测试。

    从当前交点向光源发射一条 Shadow Ray。
    如果 Shadow Ray 在到达光源之前撞到其他物体，
    则说明该点处于阴影中。

    为避免光线与自身表面再次相交，需要将光线起点沿法线方向
    偏移一个极小值 EPS。
    """
    lp = light_pos[None]

    to_light = lp - p
    dist_to_light = ti.sqrt(to_light.dot(to_light))
    light_dir = to_light / dist_to_light

    # 核心：解决 Shadow Acne
    shadow_origin = p + n * EPS

    t, hp, hn, mid = scene_intersect(shadow_origin, light_dir)

    in_shadow = False

    if t < dist_to_light:
        in_shadow = True

    return in_shadow, light_dir, dist_to_light


@ti.func
def shade_diffuse(p, n, base_color):
    """
    漫反射材质着色。

    使用简单 Lambert 光照模型：
        diffuse = base_color * max(dot(N, L), 0)

    如果当前点在阴影中，则只保留环境光。
    """
    ambient = 0.12
    color = base_color * ambient

    in_shadow, light_dir, light_dist = shadow_test(p, n)

    if not in_shadow:
        ndotl = ti.max(n.dot(light_dir), 0.0)

        # 简单距离衰减，让画面更自然
        attenuation = 1.0 / (0.10 * light_dist * light_dist + 1.0)

        diffuse = base_color * ndotl * attenuation * 4.0
        color += diffuse

    return color


# ============================================================
# 六、任务 2：基于迭代的光线弹射
# ============================================================

@ti.func
def trace(ro, rd):
    """
    Whitted-Style 光线追踪主函数。

    传统光线追踪常使用递归：
        ray -> hit mirror -> reflected ray -> hit object ...

    但 GPU 不适合递归，因此本实验使用 for 循环进行迭代式弹射。

    final_color：
        最终颜色

    throughput：
        光线吞吐量，也可以理解为每次反射后的能量衰减系数。
    """
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    cur_ro = ro
    cur_rd = rd

    for bounce in range(8):
        if bounce < max_bounces[None]:
            t, p, n, mat_id = scene_intersect(cur_ro, cur_rd)

            # 没有击中物体，返回天空背景色
            if mat_id == 0:
                sky = ti.Vector([0.05, 0.12, 0.15])
                final_color += throughput * sky
                break

            # 红色漫反射球：计算颜色后停止弹射
            if mat_id == MAT_DIFFUSE_RED:
                base = ti.Vector([0.85, 0.12, 0.08])
                c = shade_diffuse(p, n, base)
                final_color += throughput * c
                break

            # 棋盘地面：计算棋盘颜色后停止弹射
            elif mat_id == MAT_GROUND:
                base = checker_color(p)
                c = shade_diffuse(p, n, base)
                final_color += throughput * c
                break

            # 镜面球：根据反射公式生成新的反射光线
            elif mat_id == MAT_MIRROR:
                reflectivity = 0.85

                throughput *= ti.Vector([
                    reflectivity,
                    reflectivity,
                    reflectivity
                ])

                cur_ro = p + n * EPS
                cur_rd = normalize(reflect(cur_rd, n))

    return final_color


# ============================================================
# 七、选做：MSAA 抗锯齿
# ============================================================

@ti.kernel
def render():
    """
    渲染函数。

    对每个像素发射主光线并计算颜色。

    MSAA 思路：
        在一个像素内部随机采样多条光线，
        将多条光线的颜色求平均，
        从而减少球体边缘和棋盘格边缘的锯齿。
    """
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

        # Gamma Correction，让画面亮度更接近人眼观察效果
        color = ti.Vector([
            ti.sqrt(clamp01(color.x)),
            ti.sqrt(clamp01(color.y)),
            ti.sqrt(clamp01(color.z))
        ])

        pixels[i, j] = color


# ============================================================
# 八、任务 4：UI 交互面板
# ============================================================

def main():
    """
    主函数。

    使用 ti.ui.Window 创建窗口。
    使用 gui.sub_window 创建右上角控制面板。

    可调参数：
        Light X / Light Y / Light Z：控制点光源位置
        Max Bounces：控制最大弹射次数
        Samples：控制 MSAA 采样次数
    """
    light_x = 2.5
    light_y = 4.0
    light_z = 3.5

    bounce_count = 3
    spp = 4

    light_pos[None] = ti.Vector([light_x, light_y, light_z])
    max_bounces[None] = bounce_count
    samples_per_pixel[None] = spp

    window = ti.ui.Window(
        "Ray Tracing Demo",
        (WIDTH, HEIGHT),
        vsync=False
    )

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