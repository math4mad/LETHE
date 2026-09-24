# -*- coding: utf-8 -*-
# E-图志 乙案 v4 · Blender 程序化地形志 (headless) · 打磨版
# 底料: cora-atlas/docs/figs/topos-iso.svg (19 点 24 边, 逆等距投影还原真坐标)
# 审美对表: scene1 平面矢量学院风 (圆台+金晕+双色树环), 色板一律用园子在册 hex
import math, os, sys, json, random
import bpy
from mathutils import Vector

OUT = "/tmp/topos_blender_raw.png"
P = 0.62
Z = {"git": 0.115, "sea": 0.085, "lane": 0.055}
W = {"git": 0.085, "sea": 0.055, "lane": 0.028}

N = {
 "CHORA":(560,270,"polity","域体 · 总账","chora"), "ATLAS":(80,390,"polity","图谱 · 索引","atlas"),
 "ACADEMY":(944,430,"polity","学园","akademia"), "ARCHIVE":(752,590,"polity","档房","archeion"),
 "FieldTrials":(464,230,"field","实验田","peiratos"), "SkillsRegistry":(944,270,"steles","技艺碑","technai"),
 "PostHouse":(752,350,"house","驿站","stathmos"), "CodeOfLaw":(464,390,"stele","法典碑","nomos"),
 "Ledger":(176,350,"desk","台账","logos"), "GlossarySteles":(-16,350,"steles","术语碑林","glossai"),
 "ItineraryNotes":(80,470,"desk","行记","hodos"), "NameRegister":(80,550,"desk","名册","onomata"),
 "MarketStele":(464,550,"market","市集 · 碑场","agora"), "CouriersQueue":(944,510,"depot","信使棚","angeliai"),
 "ForgeKaggle":(1328,430,"forge","炉坊","kaggle"), "MachineA":(1136,590,"machine","甲机","mache A"),
 "MachineB":(752,670,"machine","乙机","mache B"), "BookMountain":(656,790,"mountain","书山","biblos"),
 "QuarantinePort":(944,750,"port","检疫港","limen"),
}
E = [
 ("CHORA",(464,230),"FieldTrials","lane"), ("CHORA",(752,350),"SkillsRegistry","lane"),
 ("CHORA",(752,350),"PostHouse","lane"), ("ATLAS",(176,350),"Ledger","lane"),
 ("ATLAS",(-16,350),"GlossarySteles","lane"), ("ATLAS",(176,430),"ItineraryNotes","lane"),
 ("ATLAS",(272,470),"NameRegister","lane"), ("ATLAS",(272,470),"CodeOfLaw","lane"),
 ("CHORA",(656,310),"CodeOfLaw","lane"), ("MarketStele",(272,470),"CodeOfLaw","lane"),
 ("MarketStele",(272,470),"NameRegister","lane"),
 ("ACADEMY",(1040,470),"ARCHIVE","git"), ("ACADEMY",(848,390),"MarketStele","git"),
 ("ACADEMY",(1232,550),"MachineA","git"), ("ACADEMY",(1136,510),"MachineB","git"),
 ("ARCHIVE",(944,670),"MachineA","lane"), ("ARCHIVE",(1040,710),"QuarantinePort","lane"),
 ("MachineA",(1040,550),"ForgeKaggle","sea"), ("MachineB",(752,670),"ForgeKaggle","sea"),
 ("MachineB",(944,750),"QuarantinePort","lane"), ("BookMountain",(752,830),"QuarantinePort","lane"),
 ("CouriersQueue",(1040,550),"ForgeKaggle","git"), ("CouriersQueue",(944,510),"ARCHIVE","lane"),
 ("ACADEMY",(1040,470),"CouriersQueue","git"),
]
def iso(x, y):
    u = (x/96.0 + y/40.0)/2.0; v = (x/96.0 - y/40.0)/2.0
    return Vector((u*P, -v*P, 0))
POS = {k: iso(v[0], v[1]) for k, v in N.items()}

sc = bpy.context.scene
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()

# 园子在册色板 (按 sRGB 书写, 统一转线性入渲染器)
COL = {
 "parch":(0.965,0.940,0.870),   # #FBF7EC 奶油
 "ink":(0.184,0.165,0.141),     # #2F2A24 墨
 "gold":(0.690,0.463,0.165),    # #B0762A 暖金 (大路)
 "gold2":(0.835,0.639,0.263),   # 亮金 (晕环/顶饰)
 "brick":(0.549,0.184,0.184),   # #8C2F2F 砖红
 "teal":(0.180,0.561,0.639),    # #2E8FA3 湖青 (海路)
 "lane":(0.780,0.720,0.570),    # 仓内巷
 "mint":(0.553,0.722,0.475),    # 薄荷
 "pine":(0.420,0.620,0.380),    # 深松绿
 "slate":(0.420,0.440,0.460),
 "fire":(1.0,0.55,0.15),
 "pond":(0.620,0.830,0.900),
 "cream":(0.990,0.980,0.950),
}
COL = {k: tuple((max(0.0, (c+0.055)/1.055))**2.4 if c > 0.04045 else c/12.92 for c in v) for k, v in COL.items()}
def mat(name, col):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if b: b.inputs["Base Color"].default_value = (*col, 1)
    return m
M = {k: mat("m_"+k, c) for k, c in COL.items()}
def box(name, loc, size, material, rz=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(*loc[:2], loc[2]+size[2]/2.0))
    o = bpy.context.object; o.name = name; o.scale = list(size)
    o.rotation_euler = (0, 0, rz); o.data.materials.append(material); return o
def cyl(name, loc, r, h, material, verts=24, rz=0.0, sy=1.0, sx=1.0):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=tuple(loc), vertices=verts)
    o = bpy.context.object; o.name = name; o.rotation_euler = (0, 0, rz); o.scale = (sx, sy, 1)
    o.data.materials.append(material); return o
def cone(name, loc, r, h, material, verts=20, rz=0.0):
    bpy.ops.mesh.primitive_cone_add(radius1=r, depth=h, location=(*loc[:2], loc[2]+h/2.0), vertices=verts)
    o = bpy.context.object; o.name = name; o.rotation_euler = (0, 0, rz); o.data.materials.append(material); return o
def sph(name, loc, r, material, sx=1.0, sy=1.0, sz=1.0):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=tuple(loc), segments=16, ring_count=8)
    o = bpy.context.object; o.name = name; o.scale = (sx, sy, sz); o.data.materials.append(material); return o
def torus(name, loc, R, r, material):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, location=tuple(loc))
    o = bpy.context.object; o.name = name; o.data.materials.append(material); return o

xs = [p.x for p in POS.values()]; ys = [p.y for p in POS.values()]
cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
RPLATE = max(max(xs)-cx, cx-min(xs), max(ys)-cy, cy-min(ys)) + 1.55

# ---------- 圆台 + 金晕 ----------
cyl("plate", (cx, cy, -0.11), RPLATE, 0.22, M["parch"], verts=96)
torus("halo", (cx, cy, 0.005), RPLATE, 0.045, M["gold2"])
cyl("plate_rim", (cx, cy, -0.13), RPLATE+0.07, 0.06, M["gold"], verts=96)

# ---------- 道路 ----------
def road(pts, kind, zz=None):
    cu = bpy.data.curves.new("rc", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts)-1)
    for p, pt in zip(sp.points, pts): p.co = (pt.x, pt.y, (zz if zz is not None else Z[kind]), 1)
    ob = bpy.data.objects.new("road_"+kind, cu); sc.collection.objects.link(ob)
    cu.bevel_depth = W[kind]; cu.bevel_resolution = 2; cu.use_fill_caps = True
    ob.data.materials.append({"git": M["gold"], "sea": M["teal"], "lane": M["lane"]}[kind]); return ob
for a, bend, b, kind in E:
    A, B = POS[a], POS[b]; Bp = iso(*bend)
    pts = [A, Bp] if (Bp - B).length < 0.05 else [A, Bp, B]
    road(pts, kind)
    if kind == "git":
        d = (pts[-1]-pts[0]); n = Vector((-d.y, d.x, 0)); n.normalize(); n *= 0.15
        road([p+n for p in pts], "lane", zz=Z[kind]); road([p-n for p in pts], "lane", zz=Z[kind])

# ---------- 建筑点缀 ----------
def temple(loc, s=1.0):
    box("t_sty", (loc.x, loc.y, loc.z), (1.15*s, 0.95*s, 0.055), M["cream"])
    box("t_sty2", (loc.x, loc.y, loc.z+0.055), (1.0*s, 0.8*s, 0.045), M["cream"])
    for k in range(8):
        fx = 0.44*s if k % 2 == 0 else 0.30*s
        fz = 0.30*s if k % 2 == 0 else 0.42*s
        cyl("t_col", (loc.x+fx, loc.y+fz, loc.z+0.10), 0.042*s, 0.34*s, M["cream"])
        cyl("t_col", (loc.x-fx, loc.y-fz, loc.z+0.10), 0.042*s, 0.34*s, M["cream"])
        cyl("t_col", (loc.x+fx, loc.y-fz, loc.z+0.10), 0.042*s, 0.34*s, M["cream"])
        cyl("t_col", (loc.x-fx, loc.y+fz, loc.z+0.10), 0.042*s, 0.34*s, M["cream"])
    box("t_ent", (loc.x, loc.y, loc.z+0.44), (1.05*s, 0.85*s, 0.075), M["brick"])
    box("t_cel", (loc.x, loc.y, loc.z+0.515), (0.80*s, 0.62*s, 0.20), M["cream"])
    cone("t_roof", (loc.x, loc.y, loc.z+0.715), 0.62*s, 0.30*s, M["brick"], verts=4, rz=math.radians(45))
def polity(loc):
    box("p_st1", loc, (0.62, 0.62, 0.05), M["cream"])
    box("p_st2", (loc.x, loc.y, loc.z+0.05), (0.48, 0.48, 0.045), M["cream"])
    st = box("p_ste", (loc.x, loc.y, loc.z+0.095), (0.13, 0.13, 0.62), M["brick"], rz=math.radians(45))
    st.scale = (0.13, 0.09, 0.62)
    torus("p_ring", (loc.x, loc.y, loc.z+0.76), 0.085, 0.022, M["gold2"])
def prop(loc, kind):
    if kind == "field":
        box("f_base", loc, (1.05, 0.85, 0.03), M["cream"])
        for i in range(3):
            for j in range(2):
                box("f_d", (loc.x-0.32+i*0.32, loc.y-0.2+j*0.4, loc.z+0.03), (0.26, 0.30, 0.028),
                    M["mint"] if (i+j) % 2 == 0 else M["pine"])
        for (dx, dy) in ((-0.45,-0.35),(0.45,-0.35),(-0.45,0.35),(0.45,0.35)):
            cone("f_t", (loc.x+dx, loc.y+dy, loc.z+0.03), 0.07, 0.20, M["pine"], verts=12)
    elif kind == "steles":
        for i, h in enumerate((0.42, 0.55, 0.42)):
            cyl("s_c", (loc.x+(i-1)*0.21, loc.y, loc.z), 0.062, h, M["cream"], 8)
            sph("s_cap", (loc.x+(i-1)*0.21, loc.y, loc.z+h), 0.062, M["cream"], sz=0.6)
    elif kind == "stele":
        b = box("c_l", loc, (0.30, 0.09, 0.52), M["brick"], rz=math.radians(18))
        sph("c_top", (loc.x+0.0, loc.y, loc.z+0.54), 0.10, M["gold2"], sz=0.7)
    elif kind == "desk":
        cyl("d_p", loc, 0.21, 0.15, M["cream"], 20)
        cyl("d_p2", (loc.x, loc.y, loc.z+0.15), 0.15, 0.05, M["slate"], 20)
        box("d_b", (loc.x, loc.y, loc.z+0.20), (0.22, 0.15, 0.045), M["parch"])
    elif kind == "market":
        cyl("m_r", (loc.x, loc.y, loc.z), 0.50, 0.035, M["cream"], 32)
        for k2 in range(4):
            a = k2*math.pi/2 + math.pi/4
            cyl("m_po", (loc.x+0.33*math.cos(a), loc.y+0.33*math.sin(a), loc.z), 0.022, 0.34, M["slate"])
        for k2 in range(6):
            box("m_s", (loc.x-0.42+k2*0.17, loc.y, loc.z+0.36), (0.085, 0.86, 0.02),
                M["gold2"] if k2 % 2 == 0 else M["cream"], rz=math.radians(6))
    elif kind == "house":
        box("h_st", loc, (0.44, 0.36, 0.035), M["cream"])
        box("h_b", (loc.x, loc.y, loc.z+0.035), (0.34, 0.27, 0.20), M["cream"])
        cone("h_r", (loc.x, loc.y, loc.z+0.235), 0.27, 0.15, M["brick"], verts=4)
    elif kind == "depot":
        box("q_b", loc, (0.40, 0.26, 0.15), M["slate"])
        cone("q_r", (loc.x, loc.y, loc.z+0.15), 0.24, 0.10, M["brick"], verts=4)
        cyl("q_f", (loc.x+0.24, loc.y, loc.z), 0.014, 0.42, M["cream"])
        box("q_fl", (loc.x+0.30, loc.y, loc.z+0.34), (0.11, 0.012, 0.07), M["brick"])
    elif kind == "forge":
        cyl("g_c", loc, 0.27, 0.32, M["slate"], 24)
        cyl("g_mouth", (loc.x, loc.y-0.24, loc.z+0.10), 0.075, 0.10, M["ink"], 12, sy=0.5)
        cone("g_f", (loc.x, loc.y, loc.z+0.32), 0.14, 0.30, M["fire"], verts=12)
    elif kind == "machine":
        box("mc", loc, (0.30, 0.26, 0.15), M["slate"])
        box("mc2", (loc.x, loc.y, loc.z+0.15), (0.24, 0.20, 0.05), M["ink"])
        box("ml", (loc.x, loc.y-0.105, loc.z+0.15), (0.16, 0.012, 0.030), M["teal"])
        cyl("ma", (loc.x+0.12, loc.y+0.10, loc.z+0.20), 0.010, 0.16, M["cream"])
    elif kind == "mountain":
        cone("bk1", (loc.x-0.15, loc.y+0.15, loc.z), 0.46, 0.55, M["gold2"], verts=5)
        cone("bk2", (loc.x+0.30, loc.y-0.05, loc.z), 0.30, 0.40, M["gold"], verts=5)
        cols = [(M["brick"], 0.0), (M["gold2"], 0.055), (M["cream"], 0.11)]
        for i, (mm, dz) in enumerate(cols):
            box("bk_b", (loc.x+0.05, loc.y-0.32, loc.z+dz), (0.44-i*0.04, 0.30-i*0.03, 0.055), mm, rz=math.radians(8*i-8))
    elif kind == "port":
        cyl("pt_pond", (loc.x, loc.y, loc.z-0.01), 0.55, 0.045, M["pond"], 32)
        hull = sph("pt_hull", (loc.x, loc.y, loc.z+0.06), 0.20, M["brick"], sx=1.5, sy=0.62, sz=0.34)
        cyl("pt_mast", (loc.x, loc.y, loc.z+0.10), 0.014, 0.34, M["cream"])
        box("pt_sail", (loc.x+0.06, loc.y, loc.z+0.20), (0.10, 0.008, 0.16), M["cream"], rz=math.radians(-6))
        cyl("pt_ring", (loc.x, loc.y, loc.z+0.44), 0.05, 0.012, M["ink"], 12)
for k, p in POS.items():
    kind = N[k][2]
    if kind == "polity":
        if k not in ("ACADEMY", "ARCHIVE"): polity(p)   # 双殿以庙为身, 不再立碑
    else: prop(p, kind)
temple(POS["ACADEMY"], 1.15); temple(POS["ARCHIVE"], 0.92)
# 圆台树环: 圆冠+尖松交替
random.seed(7)
for i in range(18):
    a = i*2*math.pi/18 + random.uniform(-.06,.06); r = RPLATE - 0.28
    tx, ty = cx + r*math.cos(a), cy + r*math.sin(a)
    cyl("trunk", (tx, ty, 0.0), 0.032, 0.20, M["slate"])
    if i % 2 == 0:
        sph("leaf", (tx, ty, 0.28), 0.16, M["mint"], sz=0.95)
    else:
        cone("leaf", (tx, ty, 0.20), 0.13, 0.42, M["pine"], verts=12)

# ---------- 相机 / 灯光 / 渲染 ----------
cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"; cam_data.ortho_scale = 13.2
cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam)
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = (cx, cy, 0)
cam.location = (cx+12.5, cy-12.5, 11.5)
cam.constraints.new('TRACK_TO')
cam.constraints[0].target = tgt
cam.constraints[0].track_axis = 'TRACK_NEGATIVE_Z'; cam.constraints[0].up_axis = 'UP_Y'
sc.camera = cam
bpy.ops.object.light_add(type="SUN", location=(cx+6, cy-8, 12))
sun = bpy.context.object; sun.data.energy = 2.0; sun.rotation_euler = (math.radians(38), math.radians(-12), math.radians(30))
sc.render.engine = "BLENDER_WORKBENCH"
sc.view_settings.view_transform = "Standard"
sc.display.shading.light = "FLAT"; sc.display.shading.color_type = "OBJECT"
sc.display.shading.show_shadows = False; sc.display.shading.show_cavity = True
for o in bpy.data.objects:
    if o.type in ('MESH', 'CURVE') and o.data.materials:
        key = o.data.materials[0].name.replace("m_", "")
        o.color = (*COL.get(key, (0.8, 0.8, 0.8)), 1)
sc.render.resolution_x = 1600; sc.render.resolution_y = 1100
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
bpy.context.view_layer.update(); sc.frame_set(sc.frame_current)
from bpy_extras.object_utils import world_to_camera_view as w2c
PIX = {}
for k, p in POS.items():
    h = 1.30 if N[k][2] == "polity" else 0.85
    ndc = w2c(sc, cam, Vector((p.x, p.y, p.z + h)))
    PIX[k] = [round(ndc.x*1600, 1), round((1-ndc.y)*1100, 1), N[k][3], N[k][4], N[k][2]]
json.dump(PIX, open(OUT + ".labels.json", "w"), ensure_ascii=False)
print("TOPOS-RENDER-DONE", OUT)
