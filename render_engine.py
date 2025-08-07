"""
OpenGL 3D渲染引擎
使用pygame和PyOpenGL实现3D粒子和球形渲染
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time

class RenderEngine:
    def __init__(self, width=1200, height=800, title="Hand Gesture Particle Sphere"):
        self.width = width
        self.height = height
        self.title = title
        
        # 初始化pygame和OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption(title)
        
        # 设置OpenGL
        self.setup_opengl()
        
        # 相机参数
        self.camera_pos = [0, 0, 8]
        self.camera_target = [0, 0, 0]
        self.camera_up = [0, 1, 0]
        
        # 鼠标控制
        self.mouse_sensitivity = 0.005
        self.camera_yaw = 0
        self.camera_pitch = 0
        
        # 时间
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        
        # 渲染状态
        self.is_running = True
        
        # 创建显示列表
        self.sphere_display_list = None
        self.create_sphere_display_list()
        
    def setup_opengl(self):
        """设置OpenGL参数"""
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # 启用混合（透明度）
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 启用点精灵
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        
        # 设置视口
        glViewport(0, 0, self.width, self.height)
        
        # 设置投影矩阵
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 100.0)
        
        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        
        # 设置光照
        self.setup_lighting()
        
    def setup_lighting(self):
        """设置光照"""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # 环境光
        ambient_light = [0.2, 0.2, 0.2, 1.0]
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        
        # 漫反射光
        diffuse_light = [0.8, 0.8, 0.8, 1.0]
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        
        # 镜面反射光
        specular_light = [1.0, 1.0, 1.0, 1.0]
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
        
        # 光源位置
        light_position = [5.0, 5.0, 5.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        
    def create_sphere_display_list(self):
        """创建球体显示列表"""
        self.sphere_display_list = glGenLists(1)
        glNewList(self.sphere_display_list, GL_COMPILE)
        
        # 创建球体
        radius = 1.0
        slices = 32
        stacks = 16
        
        for i in range(stacks):
            lat1 = (math.pi * (-0.5 + float(i) / stacks))
            lat2 = (math.pi * (-0.5 + float(i + 1) / stacks))
            
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                
                # 第一个点
                x1 = math.cos(lat1) * math.cos(lng)
                y1 = math.sin(lat1)
                z1 = math.cos(lat1) * math.sin(lng)
                
                # 第二个点
                x2 = math.cos(lat2) * math.cos(lng)
                y2 = math.sin(lat2)
                z2 = math.cos(lat2) * math.sin(lng)
                
                # 法线和顶点
                glNormal3f(x1, y1, z1)
                glVertex3f(radius * x1, radius * y1, radius * z1)
                
                glNormal3f(x2, y2, z2)
                glVertex3f(radius * x2, radius * y2, radius * z2)
            
            glEnd()
        
        glEndList()
    
    def update_camera(self):
        """更新相机"""
        # 鼠标控制
        mouse_rel = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:  # 左键拖拽
            self.camera_yaw += mouse_rel[0] * self.mouse_sensitivity
            self.camera_pitch -= mouse_rel[1] * self.mouse_sensitivity
            self.camera_pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.camera_pitch))
        
        # 计算相机位置
        distance = 8.0
        self.camera_pos[0] = distance * math.cos(self.camera_pitch) * math.cos(self.camera_yaw)
        self.camera_pos[1] = distance * math.sin(self.camera_pitch)
        self.camera_pos[2] = distance * math.cos(self.camera_pitch) * math.sin(self.camera_yaw)
        
        # 设置相机
        glLoadIdentity()
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            self.camera_up[0], self.camera_up[1], self.camera_up[2]
        )
    
    def clear_screen(self):
        """清屏"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.05, 0.05, 0.1, 1.0)  # 深蓝色背景
    
    def render_particles(self, particles, colors=None, sizes=None):
        """渲染粒子"""
        if not particles or len(particles) == 0:
            return
        
        # 禁用光照以获得更好的粒子效果
        glDisable(GL_LIGHTING)
        
        # 启用点精灵
        glEnable(GL_POINT_SPRITE)
        glPointSize(5.0)
        
        glBegin(GL_POINTS)
        
        particle_count = len(particles) // 3
        for i in range(particle_count):
            idx = i * 3
            if idx + 2 < len(particles):
                x, y, z = particles[idx], particles[idx + 1], particles[idx + 2]
                
                # 设置颜色
                if colors and len(colors) > i * 4 + 3:
                    color_idx = i * 4
                    glColor4f(colors[color_idx], colors[color_idx + 1], 
                             colors[color_idx + 2], colors[color_idx + 3])
                else:
                    # 默认渐变颜色
                    distance = math.sqrt(x*x + y*y + z*z)
                    hue = (distance * 0.2 + time.time() * 0.5) % 1.0
                    r, g, b = self.hsv_to_rgb(hue, 0.8, 0.9)
                    glColor4f(r, g, b, 0.8)
                
                glVertex3f(x, y, z)
        
        glEnd()
        
        glDisable(GL_POINT_SPRITE)
        glEnable(GL_LIGHTING)
    
    def render_sphere(self, radius=2.0, deformation=0.0, rotation=[0, 0, 0], 
                     color=[0.3, 0.6, 1.0], transparency=0.3):
        """渲染主球体"""
        glPushMatrix()
        
        # 应用变换
        glRotatef(rotation[0], 1, 0, 0)
        glRotatef(rotation[1], 0, 1, 0) 
        glRotatef(rotation[2], 0, 0, 1)
        glScalef(radius, radius, radius)
        
        # 设置材质
        ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, transparency]
        diffuse = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, transparency]
        specular = [1.0, 1.0, 1.0, transparency]
        shininess = 50.0
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)
        
        # 渲染球体
        if transparency < 1.0:
            glEnable(GL_BLEND)
            glDepthMask(GL_FALSE)  # 禁用深度写入以正确渲染透明物体
        
        glCallList(self.sphere_display_list)
        
        if transparency < 1.0:
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)
        
        glPopMatrix()
    
    def render_wireframe_sphere(self, radius=2.0, rotation=[0, 0, 0], 
                               color=[0.5, 1.0, 0.5], line_width=2.0):
        """渲染线框球体"""
        glPushMatrix()
        
        glDisable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(line_width)
        glColor3f(color[0], color[1], color[2])
        
        glRotatef(rotation[0], 1, 0, 0)
        glRotatef(rotation[1], 0, 1, 0)
        glRotatef(rotation[2], 0, 0, 1)
        glScalef(radius, radius, radius)
        
        glCallList(self.sphere_display_list)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
    
    def render_text_info(self, text_lines):
        """渲染文本信息（简单版本）"""
        # 注意：这里是一个简化版本，实际文本渲染需要更复杂的实现
        # 可以使用pygame的字体渲染后作为纹理贴到OpenGL中
        pass
    
    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB"""
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                    return False
                elif event.key == pygame.K_r:
                    # 重置相机
                    self.camera_yaw = 0
                    self.camera_pitch = 0
        
        return True
    
    def present(self):
        """呈现画面"""
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
    
    def get_current_time(self):
        """获取当前时间"""
        return time.time() - self.start_time
    
    def cleanup(self):
        """清理资源"""
        if self.sphere_display_list:
            glDeleteLists(self.sphere_display_list, 1)
        pygame.quit()

if __name__ == "__main__":
    # 测试渲染引擎
    engine = RenderEngine()
    
    # 创建测试粒子
    test_particles = []
    for i in range(1000):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        r = 3.0 + np.random.uniform(-0.5, 0.5)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        test_particles.extend([x, y, z])
    
    rotation = [0, 0, 0]
    
    print("渲染引擎测试启动！按ESC退出，拖拽鼠标旋转视角，R键重置")
    
    while engine.is_running:
        if not engine.handle_events():
            break
        
        engine.clear_screen()
        engine.update_camera()
        
        # 渲染测试内容
        current_time = engine.get_current_time()
        
        # 旋转球体
        rotation[0] = current_time * 10
        rotation[1] = current_time * 15
        rotation[2] = current_time * 5
        
        # 渲染主球体
        engine.render_sphere(
            radius=2.0,
            rotation=rotation,
            color=[0.3, 0.6, 1.0],
            transparency=0.3
        )
        
        # 渲染线框球体
        engine.render_wireframe_sphere(
            radius=3.0,
            rotation=[r * 0.5 for r in rotation],
            color=[0.5, 1.0, 0.5]
        )
        
        # 渲染粒子
        engine.render_particles(test_particles)
        
        engine.present()
    
    engine.cleanup()