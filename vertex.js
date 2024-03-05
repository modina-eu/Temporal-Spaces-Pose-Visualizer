function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
}

function createTransformFeedbackProgram(gl, vertexShaderSource, fragmentShaderSource, varyings) {
    const program = gl.createProgram();
    gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
    gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    gl.transformFeedbackVaryings(program, varyings, gl.SEPARATE_ATTRIBS);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
}

function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
}

function createVbo(gl, array, usage) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vbo;
}

const VERTEX_SHADER =
`#version 300 es

out vec2 o_sound;

uniform float u_blockOffset;
uniform float u_sampleRate;
uniform sampler2D u_modelTexture;

float hs(float t){return fract(sin(dot(t,45.))*7845.2326)-0.5;}
float tri(float t){ return mix(fract(t*2.),1.-fract(t*2.),step(0.5,fract(t)));}
float bt(float t){ return floor(fract(t*0.1)*4.)/4.;}

float rd(float t){ return fract(sin(dot(floor(t),45.6))*7845.236);}
float no(float t){ return mix(rd(t),rd(t+1.),smoothstep(0.,1.,fract(t)));}
float ns(float t, float a, float b){ return mix(rd(t),rd(t+1.),smoothstep(a,b,fract(t)));}
float eu( float tp, float nb, float time ){
    float t = mod(floor( time )*tp,nb);
    return floor((t-tp)/tp)+1.0+fract(time);}
vec2 r1(float t){

  float s1 = mix(6.,8.,step(0.5,fract(t*0.25)));
    float t1 = ns(t*200.,clamp(1.-pow(eu(s1,8.,t*4.),0.7),0.,1.),1. )*0.5;

     return vec2(t1,t1);
    }
void main(void) {
  float time = u_blockOffset + float(gl_VertexID) / u_sampleRate;
  o_sound =  smoothstep(0.4,0.5,texture(u_modelTexture, vec2(time/30.)).xy)*sin(time*800.)+r1(time)*0.;
}
`;

const FRAGMENT_SHADER =
`#version 300 es
void main(void) {}
`

function createAudio(modelTexture) {
    const DURATION = 180; // seconds
    const SAMPLES = 65536;

    const audioCtx = new AudioContext();
    const audioBuffer = audioCtx.createBuffer(2, audioCtx.sampleRate * DURATION, audioCtx.sampleRate);

    const canvas2 = document.getElementById('audio-canvas');
    const gl2 = canvas2.getContext('webgl2');

    const program = createTransformFeedbackProgram(gl2, VERTEX_SHADER, FRAGMENT_SHADER, ['o_sound']);
    const uniforms = getUniformLocations(gl2, program, ['u_sampleRate', 'u_blockOffset']);

    const array = new Float32Array(2 * SAMPLES);
    const vbo = createVbo(gl2, array, gl2.DYNAMIC_COPY);
    const transformFeedback = gl2.createTransformFeedback();

    const numBlocks = (audioCtx.sampleRate * DURATION) / SAMPLES;
    const outputL = audioBuffer.getChannelData(0);
    const outputR = audioBuffer.getChannelData(1);

    gl2.bindTransformFeedback(gl2.TRANSFORM_FEEDBACK, transformFeedback);
    gl2.enable(gl2.RASTERIZER_DISCARD);
    gl2.useProgram(program);
    gl2.activeTexture(gl2.TEXTURE1);
    gl2.bindTexture(gl2.TEXTURE_2D, modelTexture);
    gl2.uniform1i(uniforms['u_modelTexture'], 1);

    gl2.bindTransformFeedback(gl2.TRANSFORM_FEEDBACK, transformFeedback);
    gl2.enable(gl2.RASTERIZER_DISCARD);
    gl2.useProgram(program);
    gl2.uniform1f(uniforms['u_sampleRate'], audioCtx.sampleRate);
    for (let i = 0; i < numBlocks; i++) {
        gl2.uniform1f(uniforms['u_blockOffset'], i * SAMPLES / audioCtx.sampleRate);
        gl2.bindBufferBase(gl2.TRANSFORM_FEEDBACK_BUFFER, 0, vbo);
        gl2.beginTransformFeedback(gl2.POINTS);
        gl2.drawArrays(gl2.POINTS, 0, SAMPLES);
        gl2.endTransformFeedback();
        gl2.getBufferSubData(gl2.TRANSFORM_FEEDBACK_BUFFER, 0, array);

        for (let j = 0; j < SAMPLES; j++) {
            outputL[i * SAMPLES + j] = array[j * 2];
            outputR[i * SAMPLES + j] = array[j * 2 + 1];
        }
    }

    const node = audioCtx.createBufferSource();
    node.connect(audioCtx.destination);
    node.buffer = audioBuffer;
    node.loop = false;
    node.start(0);
}
