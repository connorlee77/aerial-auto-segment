import moderngl
import numpy as np
import pyvista as pv
from PIL import Image
import time
import startinpy
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


def get_mask_mgl(pts, I, colorize_func=None):
    '''
        Renders 3d points and labels in a forward facing image. 
        Assumes that pts are already in the camera coordinate frame.

        Args:
            pts: 4xN array of (x, y, z, label)
            I: 3x3 intrinsics matrix (corrected for distortion)
            color_map: mapping from label id to rgb color array. 
        Returns:
            image: np array image of the rendered points
     '''
        
    near = 0.1
    far = 10000
    rows = 512
    cols = 640
    A = -(near + far)
    B = near * far

    K_gl = np.array([
        [I[0, 0], 0, -I[0, 2], 0],
        [0, I[1, 1], -I[1, 2], 0],
        [0, 0, near + far, near * far],
        [0, 0, -1, 0],
    ], dtype='f4')
    NDC = glOrtho(0, cols, rows, 0, near, far)

    P_gl = NDC @ K_gl

    xyz = pts[[1, 2, 0], :].T
    xyz[:, 2] *= -1
    xyz[:, 0] *= -1

    label = pts[3, :].T
    color = colorize_func(label)

    t1 = time.time()
    cloud = pv.PolyData(xyz)
    surf = cloud.delaunay_2d()
    vertices = surf.points
    faces = surf.faces.reshape(-1, 4)
    triangles = faces[:, 1:]

    # dt = startinpy.DT()
    # dt.insert(xyz, insertionstrategy="BBox")
    # vertices = dt.points[1:]
    # faces = dt.triangles
    # triangles = faces
    t2 = time.time()
    print('delaunay time: ', t2 - t1)

    vertex_colors = color
    
    t1 = time.time()
    with moderngl.create_context(standalone=True, backend='egl') as ctx:

        prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 proj;

                in vec3 in_vert;
                in vec3 in_color;

                flat out vec3 v_color;
                out vec4 pose;

                void main() {
                    v_color = in_color;
                    pose = proj*vec4(in_vert, 1.0);
                    gl_Position = proj*vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                flat in vec3 v_color;
                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            ''',
            varyings=['pose']
        )
        
        ctx.enable(moderngl.DEPTH_TEST)
        # ctx.provoking_vertex = moderngl.LAST_VERTEX_CONVENTION

        P_gl = np.ascontiguousarray(P_gl.T)
        prog['proj'].write(P_gl)

        vertices_info = np.hstack([vertices, vertex_colors]).astype('f4')

        fbo = ctx.framebuffer(
                color_attachments=ctx.texture((640, 512), 3, samples=0),
                depth_attachment=ctx.depth_texture((640, 512), samples=0)
            )
        fbo.use()

        # fbo = ctx.simple_framebuffer((640, 512))
        # fbo.use()

        ibo = ctx.buffer(triangles.astype('i4'))
        vbo = ctx.buffer(vertices_info.astype('f4'))
        vao = ctx.vertex_array(
            prog,
            [(vbo, '3f 3f', 'in_vert', 'in_color')],
            ibo,
        )

        vao.render(moderngl.TRIANGLES)
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, 1)
    t2 = time.time()
    print('rendering time: ', t2 - t1)
    return np.asarray(image)
