"""
Omniverse Kit example extension that demonstrates how to stream video (such as RTSP) to a dynamic texture using [OpenCV VideoCapture](https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html) 
and [omni.ui.DynamicTextureProvider](https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html#byteimageprovider).

TODO:
- [x] Investigate how to perform the color space conversion and texture updates in a separate thread
    - This isn't improving performance like I might expect. After profiling, it appears we are still bottlenecked by a usd context lock
- [ ] Investigate how to avoid the color space conversion and instead use the native format of the frame provided by OpenCV
"""
import asyncio
import threading
import time
from typing import List

import carb
import carb.profiler
import cv2 as cv
import numpy as np
import omni.ext
import omni.kit.app
import omni.ui
import warp as wp
from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade

DEFAULT_STREAM_URI = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
#DEFAULT_STREAM_URI = "C:/Users/jshrake/Downloads/1080p.mp4"


def create_textured_plane_prim(
    stage: Usd.Stage, prim_path: str, texture_name: str, width: float, height: float
) -> Usd.Prim:
    """
    Creates a plane prim and an OmniPBR material with a dynamic texture for the albedo map
    """
    hw = width / 2
    hh = height / 2
    # This code is mostly copy pasted from https://graphics.pixar.com/usd/release/tut_simple_shading.html
    billboard: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    billboard.CreatePointsAttr([(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)])
    billboard.CreateFaceVertexCountsAttr([4])
    billboard.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    billboard.CreateExtentAttr([(-430, -145, 0), (430, 145, 0)])
    texCoords = UsdGeom.PrimvarsAPI(billboard).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

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


class OpenCvVideoStream:
    """
    A small abstraction around OpenCV VideoCapture and omni.ui.DynamicTextureProvider,
    making a one-to-one mapping between the two
    Resources:
    - https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
    - https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
    - https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html#omni.ui.ByteImageProvider.set_bytes_data_from_gpu
    """

    def __init__(self, name: str, stream_uri: str):
        self.name = name
        self.uri = stream_uri
        self.texture_array = None
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
    def update_texture(self):
        # Rate limit frame reads to the underlying FPS of the capture stream
        now = time.time()
        time_delta = now - self._last_read
        if time_delta < 1.0 / self.fps:
            return
        self._last_read = now

        # Read the frame
        carb.profiler.begin(0, "read")
        ret, frame = self._video_capture.read()
        carb.profiler.end(0)
        # The video may be at the end, loop by setting the frame position back to 0
        if not ret:
            self._video_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            self._last_read = time.time()
            return

        # By default, OpenCV converts the frame to BGR
        # We need to convert the frame to a texture format suitable for RTX
        # In this case, we convert to BGRA, but the full list of texture formats can be found at
        # # kit\source\extensions\omni.gpu_foundation\bindings\python\omni.gpu_foundation_factory\GpuFoundationFactoryBindingsPython.cpp
        frame: np.ndarray

        carb.profiler.begin(0, "color space conversion")
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        carb.profiler.end(0)
        height, width, channels = frame.shape

        carb.profiler.begin(0, "set_bytes_data")
        self._dynamic_texture.set_data_array(frame, [width, height, channels])
        carb.profiler.end(0)

class OmniRtspExample(omni.ext.IExt):
    def on_startup(self, ext_id):
        # stream = omni.kit.app.get_app().get_update_event_stream()
        # self._sub = stream.create_subscription_to_pop(self._update_streams, name="update")
        self._streams: List[OpenCvVideoStream] = []
        self._stream_threads: List[threading.Thread] = []
        self._stream_uri_model = omni.ui.SimpleStringModel(DEFAULT_STREAM_URI)
        self._window = omni.ui.Window("OpenCV Video Streaming Example", width=800, height=200)
        with self._window.frame:
            with omni.ui.VStack():
                omni.ui.StringField(model=self._stream_uri_model)
                omni.ui.Button("Create", clicked_fn=self._on_click_create)

    @carb.profiler.profile
    def _update_stream(self, i):
        async def loop():
            while self._running:
                await asyncio.sleep(0.001)
                self._streams[i].update_texture()
        asyncio.run(loop())

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
        carb.log_info(f"Creating video steam {stream_uri} {video_stream.width}x{video_stream.height}")
        # Create the mesh + material + shader
        model_root = UsdGeom.Xform.Define(stage, prim_path)
        Usd.ModelAPI(model_root).SetKind(Kind.Tokens.component)
        create_textured_plane_prim(stage, prim_path, image_name, video_stream.width, video_stream.height)
        # Clear the string model
        # self._stream_uri_model.set_value("")
        # Create the thread to pump the video stream
        self._running = True
        i = len(self._streams) - 1
        thread = threading.Thread(target=self._update_stream, args=(i, ))
        thread.daemon = True
        thread.start()
        self._stream_threads.append(thread)

    def on_shutdown(self):
        # self._sub.unsubscribe()
        self._running = False
        for thread in self._stream_threads:
            thread.join()
        self._stream_threads = []
        self._streams = []
