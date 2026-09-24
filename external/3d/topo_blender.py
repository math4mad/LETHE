# -*- coding: utf-8 -*-
# E-图志 乙案 · Blender 程序化地形志 (headless)
# 底料: cora-atlas/docs/figs/topos-iso.svg (19 点 24 边, 逆等距投影还原真坐标)
# 构图纪律: 建筑缩为点缀, 连线拓扑为视觉主角; 双语标注 (拉丁衬线 + 小字中文) 由 PIL 合成
import math, os, sys, json, random
import bpy
from mathutils import Vector

OUT = sys.argv[-1] if len(sys.argv) > 1 else "/tmp/topos_blender_raw.png"
P = 0.62                      # 每格间距(米)
Z = {"git": 0.10, "sea": 0.07, "lane": 0.045}   # 道路抬升 = 主角分层
W = {"git": 0.085, "sea": 0.055, "lane": 0.030} # 路宽: git 大路最粗

# ---------- 底料数据 (节点: svg 坐标 + 类别; 逆投影 u=(x/96+y/40)/2, v=(x/96-y/40)/2) ----------
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
E = [ # (起, 弯点svg, 止, 线种)  —— 直边以起=弯表示
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
def iso(x, y):  # svg -> 真平面坐标 (等距逆解)
    u = (x/96.0 + y/40.0)/2.0; v = (x/96.0 - y/40.0)/2.0
    return Vector((u*P, -v*P, 0))
POS = {k: iso(v[0], v[1]) for k, v in N.items()}

# ---------- 工具 ----------
sc = bpy.context.scene
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
def mat(name, col, emis=0.0, rough=0.6):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough
    if emis > 0:
        b.inputs["Emission Color"].default_value = (*col, 1)
        b.inputs["Emission Strength"].default_value = emis
    return m
COL = {"parch":(0.99,0.96,0.89), "gold":(0.78,0.52,0.16), "teal":(0.18,0.62,0.68),
       "lane":(0.80,0.76,0.65), "red":(0.52,0.16,0.16), "cream":(0.96,0.93,0.85),
       "mint":(0.55,0.72,0.47), "brick":(0.68,0.33,0.22), "slate":(0.42,0.44,0.46),
       "fire":(1.0,0.55,0.15), "sea2":(0.66,0.78,0.86)}
M = {k: mat("m_"+k, c, 2.2 if k=="fire" else (0.9 if k in("gold","teal") else 0.0), 0.4) for k,c in COL.items()}

def box(name, loc, size, material, z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(*loc[:2], loc[2]+size[2]/2.0+z))
    o = bpy.context.object; o.name = name; o.scale = list(size)
    o.data.materials.append(material); return o
def cyl(name, loc, r, h, material, verts=24):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(*loc[:2], loc[2]+h/2.0), vertices=verts)
    o = bpy.context.object; o.name = name; o.data.materials.append(material); return o
def cone(name, loc, r, h, material, verts=20):
    bpy.ops.mesh.primitive_cone_add(radius1=r, depth=h, location=(*loc[:2], loc[2]+h/2.0), vertices=verts)
    o = bpy.context.object; o.name = name; o.data.materials.append(material); return o

# ---------- 地台 ----------
xs = [p.x for p in POS.values()]; ys = [p.y for p in POS.values()]
cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
pl = box("plate", (cx, cy, -0.20), (max(xs)-min(xs)+2.2, max(ys)-min(ys)+2.2, 0.20), M["parch"])
md = pl.modifiers.new("bev", "BEVEL"); md.width = 0.09; md.segments = 3
# ---------- 道路 (视觉主角: 抬升+自发光描边) ----------
def road(pts, kind):
    cu = bpy.data.curves.new("rc", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new("POLY"); sp.points.add(len(pts)-1)
    for p, pt in zip(sp.points, pts): p.co = (pt.x, pt.y, Z[kind], 1)
    ob = bpy.data.objects.new("road_"+kind, cu); sc.collection.objects.link(ob)
    cu.bevel_depth = W[kind]; cu.bevel_resolution = 2; cu.use_fill_caps = True
    ob.data.materials.append({"git": M["gold"], "sea": M["teal"], "lane": M["lane"]}[kind]); return ob
for a, bend, b, kind in E:
    A, B = POS[a], POS[b]; Bp = iso(*bend)
    pts = [A, Bp] if (Bp - B).length < 0.05 else [A, Bp, B]
    road(pts, kind)
    if kind == "git":  # 大路加双侧细栏, 更粗更亮
        d = (pts[-1]-pts[0]); n = Vector((-d.y, d.x, 0)); n.normalize(); n *= 0.14
        road([p+n for p in pts], "lane"); road([p-n for p in pts], "lane")

# ---------- 建筑点缀 (拇指大) ----------
def temple(loc, s=1.0):
    cyl("t_base", loc, 0.55*s, 0.06, M["cream"])
    for k in range(6):
        a = k*math.pi/3
        cyl("t_col", (loc.x+0.42*s*math.cos(a), loc.y+0.42*s*math.sin(a), loc.z+0.06), 0.045*s, 0.30*s, M["cream"])
    c = cone("t_roof", (loc.x, loc.y, loc.z+0.36), 0.55*s, 0.26*s, M["brick"], verts=4); c.rotation_euler=(0,0,math.pi/4)
for k, p in POS.items():
    kind = N[k][2]
    if kind == "polity":           # 红菱碑 (在册符号): 立式菱板
        b = box("p_"+k, p, (0.30, 0.30, 0.62), M["red"]); b.rotation_euler = (0, 0, math.radians(45))
        cyl("p_ring", (p.x, p.y, p.z+0.62), 0.10, 0.05, M["gold"])
    elif kind == "temple": pass
for hub in ("ACADEMY","ARCHIVE"):  # 双殿有柱列
    temple(POS[hub], 1.15 if hub=="ACADEMY" else 0.95)
    # 市集/实验田等小件
def prop(loc, kind):
    if kind == "field":
        for i in range(2):
            for j in range(2):
                b = box("f_d", (loc.x-0.24+i*0.48, loc.y-0.24+j*0.48, loc.z), (0.34,0.34,0.035), M["mint"])
                b.rotation_euler = (0,0,math.radians(45))
    elif kind == "steles":
        for i in range(3): cyl("s_c", (loc.x+(i-1)*0.20, loc.y, loc.z), 0.055, 0.42+ (0.08 if i==1 else 0), M["slate"], 8)
    elif kind == "stele":
        b = box("c_l", loc, (0.30, 0.09, 0.50), M["cream"]); b.rotation_euler=(0,0,math.radians(20))
    elif kind == "desk":
        cyl("d_p", loc, 0.20, 0.16, M["slate"]); box("d_b", (loc.x, loc.y, loc.z+0.16), (0.20,0.14,0.04), M["cream"])
    elif kind == "market":
        cyl("m_r", (loc.x, loc.y, loc.z), 0.46, 0.035, M["cream"])
        for k2 in range(5):
            a = k2*2*math.pi/5; cyl("m_c", (loc.x+0.34*math.cos(a), loc.y+0.34*math.sin(a), loc.z), 0.035, 0.20, M["slate"])
    elif kind == "house":
        box("h_b", loc, (0.30, 0.26, 0.20), M["cream"]); c = cone("h_r", (loc.x, loc.y, loc.z+0.20), 0.26, 0.16, M["brick"], 4); c.rotation_euler=(0,0,math.radians(45))
    elif kind == "depot":
        box("q_b", loc, (0.36, 0.22, 0.16), M["slate"]); cyl("q_f", (loc.x, loc.y+0.20, loc.z), 0.02, 0.34, M["red"]); 
    elif kind == "forge":
        cyl("g_c", loc, 0.26, 0.30, M["slate"]); cone("g_f", (loc.x, loc.y, loc.z+0.30), 0.16, 0.26, M["fire"])
    elif kind == "machine":
        b = box("mc", loc, (0.26, 0.26, 0.14), M["slate"]); box("ml", (loc.x, loc.y, loc.z+0.14), (0.05, 0.05, 0.10), M["teal"])
    elif kind == "mountain":
        cone("bk1", loc, 0.44, 0.30, M["gold"]); cone("bk2", (loc.x+0.24, loc.y-0.16, loc.z), 0.28, 0.42, M["gold"])
    elif kind == "port":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.26, location=(*loc[:2], loc.z+0.05), segments=16, ring_count=8)
        s = bpy.context.object; s.name="pt_hull"; s.scale=(1.5,0.7,0.35); s.data.materials.append(M["brick"])
        cyl("pt_mast", (loc.x, loc.y, loc.z+0.10), 0.018, 0.40, M["cream"])
for k, p in POS.items():
    if N[k][2] in ("field","steles","stele","desk","market","house","depot","forge","machine","mountain","port"):
        prop(p, N[k][2])
# 薄荷树冠环绕 (风格一血统)
random.seed(7)
hx, hy = (max(xs)-min(xs))/2 + 0.9, (max(ys)-min(ys))/2 + 0.9
for i in range(16):
    t = i/16.0 * 2*math.pi
    tx, ty = cx + hx*0.94*math.cos(t), cy + hy*0.94*math.sin(t)
    cyl("trunk", (tx, ty, 0), 0.035, 0.22, M["slate"]); cone("leaf", (tx, ty, 0.22), 0.20, 0.44, M["mint"])

# ---------- 相机 (正交等距) 与灯光 ----------
cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"; cam_data.ortho_scale = 13.4; cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam)
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = (cx, cy, 0)
cam.location = (cx+13.5, cy-13.5, 12.0)          # 东南高站
cam.constraints.new('TRACK_TO')
cam.constraints[0].target = tgt
cam.constraints[0].track_axis = 'TRACK_NEGATIVE_Z'
cam.constraints[0].up_axis = 'UP_Y'
sc.camera = cam
bpy.ops.object.light_add(type="SUN", location=(cx+6, cy-8, 12))
sun = bpy.context.object; sun.data.energy = 3.2; sun.rotation_euler = (math.radians(38), math.radians(-12), math.radians(30))
w = sc.world or bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.90, 0.82, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.65

# ---------- 渲染 (Workbench: headless 稳) ----------
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "FLAT"; sc.display.shading.color_type = "OBJECT"
for o in bpy.data.objects:
    if o.type in ('MESH', 'CURVE') and o.data.materials:
        key = o.data.materials[0].name.replace("m_", "")
        o.color = (*COL.get(key, (0.8, 0.8, 0.8)), 1)
sc.display.shading.show_shadows = True; sc.display.shading.show_cavity = True
sc.render.resolution_x = 1600; sc.render.resolution_y = 1100
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
# 节点屏幕坐标导出 (供 PIL 双语合成): [px, py, 中文, 拉丁名, 类别]
from bpy_extras.object_utils import world_to_camera_view as w2c
bpy.context.view_layer.update()
sc.frame_set(sc.frame_current)
PIX = {}
for k, p in POS.items():
    h = 1.15 if N[k][2] == "polity" else 0.8
    ndc = w2c(sc, cam, Vector((p.x, p.y, p.z + h)))
    PIX[k] = [round(ndc.x*1600, 1), round((1-ndc.y)*1100, 1), N[k][3], N[k][4], N[k][2]]
json.dump(PIX, open(OUT + ".labels.json", "w"), ensure_ascii=False)
print("TOPOS-RENDER-DONE", OUT)
