
import numpy as np
from pyrr import Matrix44, matrix44



class MeshBase(Renderable):
    """Abstract base class that implements functions commonly used by the
    subclasses."""
    def __init__(self, vertices, normals, offset=[0, 0, 0.]):
        self._vertices = np.asarray(vertices)
        self._normals = np.asarray(normals)
        self._model_matrix = np.eye(4).astype(np.float32)
        self._offset = np.asarray(offset).astype(np.float32)

        self._prog = None
        self._vbo = None
        self._vao = None

    def to_points_and_faces(self):
        N = 0
        index = {}
        faces = []
        for x in self._vertices.reshape(-1, 3):
            x = tuple(x)
            if x not in index:
                index[x] = N
                N += 1
            faces.append(index[x])
        points = [p for p, i in sorted(index.items(), key=lambda x: x[1])]

        return np.array(points), np.array(faces).reshape(-1, 3)

    @property
    def vertices(self):
        """Return all the vertices of this mesh. The vertices in groups of 3
        form triangles and several vertices occur multiple times as they are
        shared by several triangles."""
        return np.copy(self._vertices)

    @property
    def bbox(self):
        """The axis aligned bounding box of all the vertices as two
        3-dimensional arrays containing the minimum and maximum for each
        axis."""
        return [
            self._vertices.min(axis=0),
            self._vertices.max(axis=0)
        ]

    @property
    def model_matrix(self):
        """An affine transformation matrix (4x4) applied to the mesh before
        rendering. Can be changed to animate the mesh."""
        return self._model_matrix

    @model_matrix.setter
    def model_matrix(self, v):
        self._model_matrix = np.asarray(v).astype(np.float32)
        if self._prog:
            self._prog["local_model"].write(self._model_matrix.tobytes())

    def to_unit_cube(self):
        """Transform the mesh such that it fits in the 0 centered unit cube."""
        bbox = self.bbox
        dims = bbox[1] - bbox[0]
        self._vertices -= dims/2 + bbox[0]
        self._vertices /= dims.max()
        self._update_vbo()

    def release(self):
        self._prog.release()
        self._vbo.release()
        self._vao.release()
        self._prog = None
        self._vbo = None
        self._vao = None

    def render(self):
        self._vao.render()

    def update_uniforms(self, uniforms):
        uniforms_list = self._get_uniforms_list()
        for k, v in uniforms:
            if k in uniforms_list:
                self._prog[k].write(v.tobytes())

    @staticmethod
    def _triangle_normals(triangles):
        triangles = triangles.reshape(-1, 3, 3)
        ba = triangles[:, 1] - triangles[:, 0]
        bc = triangles[:, 2] - triangles[:, 1]
        return np.cross(ba, bc, axis=-1)



class Mesh(MeshBase):
    """A mesh is a collection of triangles with normals and colors.

    Arguments
    ---------
        vertices: array-like, the vertices of the triangles. Each triangle
                  should be given on its own even if vertices are shared.
        normals: array-like, per vertex normal vectors
        colors: array-like, per vertex color as (r, g, b) floats or
                (r, g, b, a) floats. If one color is given then it is assumed
                to be for all vertices.
        offset: A translation vector for all the vertices. It can be changed
                after construction to animate the object together with the
                `model_matrix` property.
        mode: {"shading", "flat", "depth", "orthographic_depth"} A string that
              defines the rendering mode for this mesh. (default: shading)
        max_depth: The maximum depth map value when rendering depth maps
    """
    def __init__(self, vertices, normals, colors, max_depth=3.0):
        super(Mesh, self).__init__(vertices, normals, offset)

        self._colors = np.asarray(colors)
        N = len(self._vertices)
        self._colors = np.hstack([self._colors, np.ones((N, 1))])
        self._max_depth = max_depth

    def init(self, ctx):
        self._prog = ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 mvp;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec4 in_color;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec4 v_color;

                void main() {
                    v_color = in_color;

                    vec4 t_position = vec4(in_vert, 1.0);
                    t_position = mvp * t_position;
                    gl_Position = t_position;

                    vec4 t_normal = vec4(in_norm, 1.0);
                    v_norm = t_normal.xyz / t_normal.w;
                }
            """,
            fragment_shader="""
                #version 330

                in vec4 v_color;
                out vec4 f_color;

                void main() {
                    f_color = v_color;
                }
            """
        )

        self._vbo = ctx.buffer(np.hstack([
            self._vertices,
            self._normals,
            self._colors
        ]).astype(np.float32).tobytes())
        self._vao = ctx.simple_vertex_array(
            self._prog,
            self._vbo,
            "in_vert", "in_norm", "in_color"
        )
        self.max_depth = self._max_depth