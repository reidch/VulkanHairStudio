# ICT FaceKit Light — default professional character surface

Upstream: https://github.com/USC-ICT/ICT-FaceKit

Default source file: `FaceXModel/generic_neutral_mesh.obj`

License: MIT. CMake downloads the upstream OBJ and LICENSE on first configure unless local paths are supplied.

V37 consumes the released neutral OBJ as a whole. It does not crop a visual or collision material subset, does not rewrite upstream face winding, and does not fabricate a substitute mannequin.

Offline configuration:

```text
-DICT_FACEKIT_LOCAL_OBJ=C:/path/to/generic_neutral_mesh.obj
-DICT_FACEKIT_LOCAL_LICENSE=C:/path/to/ICT-FaceKit/LICENSE
```
