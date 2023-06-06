import bpy
import mathutils
import numpy as np
from mathutils import Vector
import bmesh
from bpy import context

#triangulate faces
def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    return me

#create mesh from faces
def createMeshFromData(name, origin, verts, faces):
    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new(name, me)
    ob.location = origin
    ob.show_name = True
 
 
    scn = bpy.context.scene
    scn.collection.objects.link(ob)
    
    bpy.context.view_layer.objects.active=ob
    ob.select_set(True)
    #scn.collection.objects.link(ob)
    #scn.collection.objects.active = ob
    
    
 
    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me=triangulate_object(ob)
    me.update()    
    return ob

#get control points from file
def make_ControlPts(fileName):
    contrlPts=[]
    for line in open(fileName,"r"):
        if line.startswith('#'):continue
        values=line.split()
        if not values:continue
        if values[0]=='v':
            point=[]
            point.append(10*float(values[1]))
            point.append(10*float(values[2]))
            point.append(10*float(values[3]))
            contrlPts.append(point)
    return contrlPts


#de casteljau algorithm
def de_casteljau(t, points):
    pts = [p for p in points] # values in this list are overridden
    n = len(points)
    for j in range(1, n):
        for k in range(n - j):
            pts[k] = pts[k] * (1 - t) + pts[k + 1] * t
    return pts[0]

#calculate points on Bezier curves
def Bezier(pts):
    beta = []
    for t in range(0, 101, 1):
        t /=100 
        beta.append(de_casteljau(t, pts))
    return beta

#make faces based on interpolation points
def makeFaces(pts):
    faces=[]
    
    #make faces based on 4 points
    for i in range(100):
        for j in range(100):
            index = []
            index.append(101*i+j)
            index.append(101*(i+1)+j)
            index.append(101*(i+1)+j+1)
            index.append(101*i+j+1)
            faces.append(index)
    
    return faces

def make_ob_file(pts):
    faces = makeFaces(pts)
    ob=createMeshFromData("test",(0,0,0),pts,faces)
    return ob

file_path="/Users/ligang/Desktop/Assignment2/data/points.txt"
verts=make_ControlPts(file_path)
verts= np.array(verts)


curve1 = []
curve2 = []
curve3 = []
curve4 = []
#first pair of curves
curve1.extend(Bezier(verts[:4]))              #curve1
curve2.extend(Bezier(verts[4:8]))             #curve2 
#second pari of curves
curve3.extend(Bezier(verts[8:12]))            #curve3
curve4.extend(Bezier(verts[12:16]))           #curve4

curve1 = np.array(curve1)
curve2 = np.array(curve2)
curve3 = np.array(curve3)
curve4 = np.array(curve4)

mesh_pts = []

for i in range(0, 101, 1):
    for j in range(0, 101, 1):
        #interpolate 100 times
        s = i/100
        t = j/100

        plane_1 = curve1[i]*(1-t) + curve2[i]*(t)

        plane_2 = curve3[j]*(1-s) + curve4[j]*(s)

        plane_3 = curve1[0]*(1-s)*(1-t) + curve1[-1]*s*(1-t) + curve2[0]*(1-s)*t + curve2[-1]*s*t

        plane = plane_1 + plane_2 - plane_3

        mesh_pts.append(plane)
mesh_pts = np.array(mesh_pts)

ob = make_ob_file(mesh_pts)
