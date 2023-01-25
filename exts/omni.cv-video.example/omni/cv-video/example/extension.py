'''
Omniverse Kit example extension that demonstrates how to stream video (such as RTSP) to a dynamic texture using [OpenCV VideoCapture](https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html) 
and [omni.ui.DynamicTextureProvider](https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html#byteimageprovider).

TODO:
- [ ] Investigate how to perform the color space conversion and texture updates in a separate thread
- [ ] Investigate how to avoid the color space conversion and instead use the native format
'''
import omni.ext
import omni.ui
import omni.kit.app
import cv2 as cv
import numpy as np
import carb
import carb.profiler
import time
from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade
from typing import List

DEFAULT_STREAM_URI = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"

def create_textured_plane_prim(stage: Usd.Stage, prim_path: str, texture_name: str, width: float, height: float) -> Usd.Prim:
    '''
    Creates a plane prim and an OmniPBR material with a dynamic texture for the albedo map
    '''
    hw = width / 2
    hh = height / 2
    # This code is mostly copy pasted from https://graphics.pixar.com/usd/release/tut_simple_shading.html
    billboard: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    billboard.CreatePointsAttr([(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)])
    billboard.CreateFaceVertexCountsAttr([4])
    billboard.CreateFaceVertexIndicesAttr([0,1,2,3])
    billboard.CreateExtentAttr([(-430, -145, 0), (430, 145, 0)])
    texCoords = UsdGeom.PrimvarsAPI(billboard).CreatePrimvar("st",
                                        Sdf.ValueTypeNames.TexCoord2fArray,
                                        UsdGeom.Tokens.varying)
    texCoords.Set([(0, 0), (1, 0), (1,1), (0, 1)])

    material_path = f"{prim_path}/Material"
    material: UsdShade.Material = UsdShade.Material.Define(stage, material_path)
    shader: UsdShade.Shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(f"dynamic://{texture_name}")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    billboard.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(billboard).Bind(material)
    return billboard

class OpenCvVideoStream():
    '''
    A small abstraction around OpenCV VideoCapture and omni.ui.DynamicTextureProvider,
    making a one-to-one mapping between the two
    Resources:
    - https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
    - https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
    - https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html#omni.ui.ByteImageProvider.set_bytes_data_from_gpu
    '''
    def __init__(self, name: str, stream_uri: str):
        self.name = name
        self.uri = stream_uri
        try:
            # Attempt to treat the uri as an int
            # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
            stream_uri_as_int = int(stream_uri)
            self._video_capture = cv.VideoCapture(stream_uri_as_int)
        except:
            # Otherwise treat the uri as a str
            self._video_capture = cv.VideoCapture(stream_uri)
        self.fps: float = self._video_capture.get(cv.CAP_PROP_FPS)
        self.width: int = self._video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height: int = self._video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        self._dynamic_texture = omni.ui.DynamicTextureProvider(name)
        self._last_read = time.time()
        self.is_ok = self._video_capture.isOpened()
        # If this FPS is 0, set it to something sensible 
        if self.fps == 0:
            self.fps = 24
    
    @carb.profiler.profile
    def update(self):
        # Rate limit frame reads to the underlying FPS of the capture stream
        now = time.time()
        time_delta =  now - self._last_read
        if (time_delta < 1.0/self.fps):
            return
        self._last_read = now

        # Read the frame
        ret, frame = self._video_capture.read()
        if not ret:
            return

        # By default, OpenCV converts the frame to BGR 
        # We need to convert the frame to a texture format suitable for RTX
        # In this case, we convert to BGRA, but the full list of texture formats can be found at
        # # kit\source\extensions\omni.gpu_foundation\bindings\python\omni.gpu_foundation_factory\GpuFoundationFactoryBindingsPython.cpp
        frame: np.ndarray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        height, width, channels = frame.shape
        # See https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html#omni.ui.ByteImageProvider.set_bytes_data_from_gpu
        self._dynamic_texture.set_bytes_data(frame.flatten().tolist(), [width, height], omni.ui.TextureFormat.BGRA8_UNORM)

class OmniRtspExample(omni.ext.IExt):
    def on_startup(self, ext_id):
        stream = omni.kit.app.get_app().get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update, name="update")
        self._streams: List[OpenCvVideoStream] = []
        self._stream_uri_model = omni.ui.SimpleStringModel(DEFAULT_STREAM_URI)
        self._window = omni.ui.Window("OpenCV Video Streaming Example", width=800, height=200)
        with self._window.frame:
            with omni.ui.VStack():
                omni.ui.StringField(model=self._stream_uri_model)
                omni.ui.Button("Create", clicked_fn=self._on_click_create)
    
    @carb.profiler.profile
    def _on_update(self, e):
        for stream in self._streams:
            stream.update()

    def _on_click_create(self):
        name = f"Video{len(self._streams)}"
        image_name = name
        usd_context = omni.usd.get_context()
        stage: Usd.Stage = usd_context.get_stage()
        prim_path = f"/World/{name}"
        # If the prim already exists, remove it so we can create it again
        try:
            stage.RemovePrim(prim_path)
            self._streams = [stream for stream in self._streams if stream.name != image_name]
        except:
            pass
        # Create the stream
        stream_uri = self._stream_uri_model.get_value_as_string()
        video_stream = OpenCvVideoStream(image_name, stream_uri)
        if not video_stream.is_ok:
            carb.log_error(f"Error opening stream: {stream_uri}")
            return
        self._streams.append(video_stream)
        # Create the mesh + material + shader
        model_root = UsdGeom.Xform.Define(stage, prim_path)
        Usd.ModelAPI(model_root).SetKind(Kind.Tokens.component)
        create_textured_plane_prim(stage, prim_path, image_name, video_stream.width, video_stream.height)
        # Clear the string model
        #self._stream_uri_model.set_value("")

    def on_shutdown(self):
        self._sub.unsubscribe()
        self._streams = []
