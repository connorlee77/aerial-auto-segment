import moderngl

import numpy as np
from PIL import Image

import cv2
from PIL import ImageColor
import pyvista as pv

def glOrtho(left, right, bottom, top, near, far):
    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    tz = -(far + near) / (far - near)
    x = np.array([
        [2 / (right - left), 0, 0, tx],
        [0, 2 / (top - bottom), 0, ty],
        [0, 0, -2 / (far - near), tz],
        [0, 0, 0, 1],
    ], dtype='f4')
    return x

HEX_COLORS = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A',
    '#C4281B', '#ffffff', '#B39FE1', '#A8DEFF'
]
    
# K_gl = np.zeros((4,4), dtype='f4')
# K_gl[0,0] = -I[0,0]
# K_gl[1,1] = -I[1,1]
# K_gl[0,2] = (cols - I[0,2])
# K_gl[1,2] = (rows - I[1,2])
# K_gl[2,2] = A
# K_gl[2,3] = B
# K_gl[3,2] = 1

# NDC = np.zeros((4,4), dtype='f4')
# NDC[0,0] = -2 / cols
# NDC[1,1] = 2 / rows
# NDC[2,2] = -2 / (far - near)

# NDC[0,3] = 1
# NDC[1,3] = -1
# NDC[2,3] = -(far + near) / (far - near)
# NDC[3,3] = 1



def dynamic_world_color_map():
    rgb_colors = [ImageColor.getcolor(c, "RGB") for c in HEX_COLORS]
    color_map = dict(zip(list(range(0, 10)), rgb_colors))
    return color_map

def colorize_dynamic_world_label(label):
    mapping = dynamic_world_color_map()

    h = len(label)
    color_label = np.zeros((h, 3), dtype=np.uint8)
    for i in range(0, 10):
        color_label[label == i, :] = mapping[i] 
    return color_label / 255

def get_mask_mgl(pts, label):

    # Intrinsics matrix
    I_old = np.array([
        [511.03573247, 0.000000, 311.80346835], 
        [0.000000, 508.22913692, 261.56701122], 
        [0.000000, 0.000000, 1.000000]
    ])
    D = np.array([-0.43339599, 0.18974767, -0.00146426, 0.00118333, 0.000000])

    H, W = (512, 640)
    I, roi = cv2.getOptimalNewCameraMatrix(I_old, D, (W, H), 0, (W, H))
    # print(I)
    near = 0.1
    far = 10000
    rows = 512
    cols = 640
    A = -(near + far)
    B  = near * far
    
    K_gl = np.array([
        [I[0,0], 0, -I[0,2], 0],
        [0, I[1,1], -I[1,2], 0],
        [0, 0, near+far, near*far],
        [0, 0, -1, 0],
    ], dtype='f4')
    NDC = glOrtho(0, cols, rows, 0, near, far)

    P_gl = NDC @ K_gl
    # pts = np.load('outputs/thermal-10000.npy').T

    xyz = pts[[1, 2, 0], :].T
    xyz[:,2] *= -1
    xyz[:,0] *= -1

    # print("Input Label: ", label.shape)
    label = pts[3,:].T
    # print("After: ", label.shape)

    # num_vertices = xyz.shape[0]
    color = colorize_dynamic_world_label(label)
    print(color)
    # colors = np.hstack([color, np.ones(num_vertices).reshape(-1,1)])

    cloud = pv.PolyData(xyz)
    surf = cloud.delaunay_2d()
    print(surf.point_normals)
    vertex_normals = surf.point_normals
    vertices = surf.points
    vertex_colors = color

    faces = surf.faces.reshape(-1, 4)
    triangles = faces[:,1:]
    # print(faces.shape)
    # print(surf.is_all_triangles)
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(xyz)
    # pcl.colors = o3d.utility.Vector3dVector(color[:,:3])
    # pcl.estimate_normals()
    # normals = np.asarray(pcl.normals)
    # bbox = pcl.get_axis_aligned_bounding_box()
    # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl)
    # rec_mesh = rec_mesh.normalize_normals()
    # p_mesh_crop = rec_mesh.crop(bbox)
    # mesh = p_mesh_crop


    # vertices_info = np.concatenate([xyz, color], axis=1)
    # print(prog['proj'].value)
    # print(vertices_info[:,:3].shape)
    # h, w = xyz.shape

    # visual = mesh.visual


    # vertices = np.asarray(mesh.vertices)
    # vertex_normals = np.asarray(mesh.vertex_normals)
    # vertex_colors = np.asarray(mesh.vertex_colors)

    # triangles = np.asarray(mesh.triangles)
    # print(triangles)
    # print(vertex_colors.shape)
    # print(vertices.shape)
    # print(np.linalg.norm(vertex_normals, axis=1))


    # -------------------
    # CREATE CONTEXT HERE
    # -------------------

    with moderngl.create_context(standalone=True, backend='egl') as ctx:

        prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 proj;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec3 in_color;

                out vec3 v_color;
                out vec3 v_norm;
                out vec4 pose;

                void main() {
                    v_norm = in_norm;
                    v_color = in_color;
                    pose = proj*vec4(in_vert, 1.0);
                    gl_Position = proj*vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                in vec3 v_color;
                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            ''',
            varyings=['pose']
        )

        P_gl = np.ascontiguousarray(P_gl.T)
        prog['proj'].write(P_gl)
        
        vertices_info = np.hstack([vertices, vertex_normals, vertex_colors]).astype('f4')

        fbo = ctx.simple_framebuffer((640, 512))
        fbo.use()
        
        ctx.enable(moderngl.DEPTH_TEST)
        ibo = ctx.buffer(triangles.astype('i4'))
        vbo = ctx.buffer(vertices_info.astype('f4'))
        vao = ctx.vertex_array(
            prog, 
            [(vbo, '3f 3f 3f', 'in_vert', 'in_norm', 'in_color')],
            ibo,
        )
        
        # storage_buffer = ctx.buffer(reserve=397*4*4)
        # vao.transform(storage_buffer, vertices=397)
        # data = struct.unpack("1588f", storage_buffer.read())
        # for i in range(0, 397*4, 4):
        #     print("{} {} {} {}".format(*data[i:i+4]))


        vao.render(moderngl.TRIANGLES)
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, 1)
        image.save('output.png')
        print(ctx.error)
        # Return as np array for plotting over original image
    return np.asarray(image) 