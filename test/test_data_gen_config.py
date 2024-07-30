import argparse
import glfw
from OpenGL.GL import *
import numpy as np
import glm
import imgui
from imgui.integrations.glfw import GlfwRenderer
import cv2
from pdf2image import convert_from_path
from PIL import Image
import io
import json
import os

# Global variables
rotation_matrix = glm.mat4(1)
last_x, last_y = 400, 400
left_mouse_button_pressed = False
theta = 0.0
phi = 0.0
radius = 5.0
shader_params = {
    "light_pos": [100.2, -120.0, 2.0],
    "eye_pos": [0.0, 0.0, 5.0]
}
gaussian_blur_params = {
    "kernel_size": (5, 5),
    "sigma": 0
}
grain_noise_params = {
    "intensity": 0.1,
    "density": 0.5
}

def init_glfw():
    if not glfw.init():
        raise Exception("GLFW can't be initialized")
    window = glfw.create_window(800, 800, "PDF Viewer", None, None)
    glfw.set_window_pos(window, 100, 100)
    glfw.make_context_current(window)
    return window

def setup_callbacks(window):
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

def mouse_button_callback(window, button, action, mods):
    global left_mouse_button_pressed, last_x, last_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        left_mouse_button_pressed = action == glfw.PRESS
        last_x, last_y = glfw.get_cursor_pos(window)

def cursor_position_callback(window, xpos, ypos):
    global rotation_matrix, last_x, last_y
    if left_mouse_button_pressed:
        dx, dy = xpos - last_x, ypos - last_y
        sensitivity = 0.01
        rotation_x = glm.rotate(glm.mat4(1), sensitivity * dy, glm.vec3(1, 0, 0))
        rotation_y = glm.rotate(glm.mat4(1), sensitivity * dx, glm.vec3(0, 1, 0))
        rotation_matrix = rotation_y * rotation_x * rotation_matrix
        last_x, last_y = xpos, ypos

def load_pdf_as_texture(pdf_path):
    # Convert first page of PDF to image
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    img = pages[0]
    
    # Convert image to numpy array
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    
    # Create OpenGL texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_np.shape[1], img_np.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_np)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    return texture

def setup_plane():
    # A4 dimensions (in mm)
    width, height = 210, 297
    
    # Normalized to -1 to 1 range
    vertices = np.array([
        -width/2, -height/2, 0.0, 0.0, 0.0,
        width/2, -height/2, 0.0, 1.0, 0.0,
        width/2, height/2, 0.0, 1.0, 1.0,
        -width/2, height/2, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    
    # Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
    glEnableVertexAttribArray(1)
    
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    glBindVertexArray(0)
    
    return vao, len(indices)

def setup_shaders():
    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    
    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 Normal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * vec3(0.0, 0.0, 1.0);  // Assuming the plane is always facing +z
        gl_Position = projection * view * vec4(FragPos, 1.0);
        TexCoord = aTexCoord;
    }
    """
    
    fragment_shader = """
    #version 330 core
    in vec2 TexCoord;
    in vec3 FragPos;
    in vec3 Normal;
    
    out vec4 FragColor;
    
    uniform sampler2D ourTexture;
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    
    void main()
    {
        // Ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
        
        // Diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
        
        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        
        vec4 texColor = texture(ourTexture, TexCoord);
        vec3 result = (ambient + diffuse + specular) * texColor.rgb;
        FragColor = vec4(result, texColor.a);
    }
    """
    
    # Compile shaders
    vertex_shader_id = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader_id, vertex_shader)
    glCompileShader(vertex_shader_id)
    
    fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader_id, fragment_shader)
    glCompileShader(fragment_shader_id)
    
    # Link shaders
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader_id)
    glAttachShader(shader_program, fragment_shader_id)
    glLinkProgram(shader_program)
    
    # Clean up
    glDeleteShader(vertex_shader_id)
    glDeleteShader(fragment_shader_id)
    
    return shader_program

def setup_matrices(shader_program, scale, axis, angle):
    glUseProgram(shader_program)
    
    model = glm.scale(glm.mat4(1.0), glm.vec3(scale, scale, scale))
    model = glm.rotate(model, glm.radians(angle), glm.vec3(*axis))
    view = glm.lookAt(glm.vec3(*shader_params["eye_pos"]), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 800 / 800, 0.1, 100.0)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
    view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniform3fv(light_pos_loc, 1, shader_params["light_pos"])
    glUniform3fv(view_pos_loc, 1, shader_params["eye_pos"])

    return model, view, projection, model_loc, view_loc, projection_loc, light_pos_loc, view_pos_loc

def render_frame(vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, light_pos_loc, view_pos_loc):
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    view = view * rotation_matrix
    
    glUseProgram(shader_program)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniform3fv(light_pos_loc, 1, shader_params["light_pos"])
    glUniform3fv(view_pos_loc, 1, shader_params["eye_pos"])
    
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_grain_noise(image, intensity, density):
    noise = np.random.rand(*image.shape[:2])
    noise = (noise < density).astype(np.float32) * intensity
    noisy_image = image.astype(np.float32) + noise[:, :, np.newaxis]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def main(args):
    global theta, phi, radius, gaussian_blur_params, grain_noise_params
    window = init_glfw()
    setup_callbacks(window)
    
    texture = load_pdf_as_texture(args.pdf_file)
    vao, num_indices = setup_plane()
    
    shader_program = setup_shaders()
    model, view, projection, model_loc, view_loc, projection_loc, light_pos_loc, view_pos_loc = setup_matrices(shader_program, args.scale, args.axis, args.angle)
    
    glEnable(GL_DEPTH_TEST)
    
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    # Load frame configurations from JSON file
    with open(args.config_file, 'r') as f:
        frame_configs = json.load(f)
    
    for i, config in enumerate(frame_configs):
        # Update parameters based on the current config
        theta = config.get('theta', theta)
        phi = config.get('phi', phi)
        radius = config.get('radius', radius)
        gaussian_blur_params['kernel_size'] = tuple(config.get('kernel_size', gaussian_blur_params['kernel_size']))
        gaussian_blur_params['sigma'] = config.get('sigma', gaussian_blur_params['sigma'])
        grain_noise_params['intensity'] = config.get('grain_intensity', grain_noise_params['intensity'])
        grain_noise_params['density'] = config.get('grain_density', grain_noise_params['density'])
        
        # Render frame
        x = radius * glm.sin(glm.radians(theta)) * glm.cos(glm.radians(phi))
        y = radius * glm.sin(glm.radians(phi))
        z = radius * glm.cos(glm.radians(theta)) * glm.cos(glm.radians(phi))
        shader_params["eye_pos"] = [x, y, z]
        view = glm.lookAt(glm.vec3(x, y, z), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        render_frame(vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, light_pos_loc, view_pos_loc)
        
        # Capture frame
        width, height = glfw.get_framebuffer_size(window)
        image_buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(height, width, 3)
        image = cv2.flip(image, 0)  # Flip vertically
        
        # Apply filters
        image = apply_gaussian_blur(image, gaussian_blur_params['kernel_size'], gaussian_blur_params['sigma'])
        image = apply_grain_noise(image, grain_noise_params['intensity'], grain_noise_params['density'])
        
        # Save image
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        glfw.swap_buffers(window)
    
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Viewer using OpenGL")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--axis", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Axis to rotate the model along")
    parser.add_argument("--angle", type=float, default=0.0, help="Angle to rotate the model")
    parser.add_argument("--scale", type=float, default=0.005, help="Scale for OBJ model file")
    parser.add_argument("--theta", type=float, default=0.0, help="Initial theta angle for viewing")
    parser.add_argument("--phi", type=float, default=0.0, help="Initial phi angle for viewing")
    parser.add_argument("--radius", type=float, default=5.0, help="Initial radius for viewing")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=float, default=0.0, help="Sigma value for Gaussian blur")
    parser.add_argument("--config_file", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
    args = parser.parse_args()
    
    theta = args.theta
    phi = args.phi
    radius = args.radius
    gaussian_blur_params["kernel_size"] = (args.kernel_size, args.kernel_size)
    gaussian_blur_params["sigma"] = args.sigma

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)